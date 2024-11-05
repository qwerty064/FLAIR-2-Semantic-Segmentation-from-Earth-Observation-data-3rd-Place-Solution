import numpy as np
import pandas as pd
from tqdm import tqdm  
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base.initialization import initialize_decoder
from torch.utils.data import Dataset 
import timm
import segmentation_models_pytorch as smp
import torchvision.transforms
import rasterio
from tta import TTA
import cv2
import json
from PIL import Image 
from pathlib import Path


class Cfg:
    tta                     = False
    num_tta                 = 4 # 4, 8
    weights                 = None 
    pretrained              = False
    batch_size              = 16
    img_size                = 512
    num_classes             = 13
    dropout                 = 0.0

    device                  = "cuda"
    use_amp                 = True 
    compile                 = False
    exp_name                = ""    
    num_workers             = 4

    # (model_name, model_path, decoder_ch_coef, model_weight)
    models_data             = [
                               ("maxvit_tiny_tf_512.in1k", "maxvit_tiny_tf_512.in1k_id0_epoch10.pt", 2, 1), 
                               ("maxvit_tiny_tf_512.in1k", "maxvit_tiny_tf_512.in1k_id1_epoch10.pt", 4, 1), 
                               ("maxvit_base_tf_512.in21k_ft_in1k", "maxvit_base_tf_512.in21k_ft_in1k_id2_epoch10.pt", 4, 2.0), 
                               ("maxvit_tiny_tf_512.in1k", "maxvit_tiny_tf_512.in1k_id3_epoch10.pt", 4, 1),  
                               ("maxvit_base_tf_512.in21k_ft_in1k", "maxvit_base_tf_512.in21k_ft_in1k_id7_epoch4.pt", 1, 1.5), 
                               ("maxvit_rmlp_pico_rw_256.sw_in1k", "pico_sentinel.pt", 1, 0.1),
                              ]

    output_dir              = "test_preds" 


class NetDataset(Dataset):
    def __init__(self, df, cfg, transform, centroid):
        super().__init__()
        self.cfg = cfg
        self.df = np.array(df)
        self.transform = transform
        self.centroids = centroid
        self.normalize_image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        data = self.df[i]
        img_path, img_id = data[0], data[1]

        with rasterio.open(img_path) as img:
            image = np.array(img.read()[:3]) 

        # sen_image
        ##################################
        parts = img_path.split('/')
        sen_img_path = "flair_2_sen_test/" + parts[-4] +'/'+ parts[-3] + "/sen/SEN2_sp_" + parts[-4]+'-'+parts[-3]+"_data.npy"

        x = np.load(sen_img_path)
        indices = np.random.randint(x.shape[0], size=16)
        x = x[indices]

        patch_size = 64 # 40-112
        a, b = self.centroids[parts[-1]]
        x = x[:,:3,::] # BGR
        x = x[:,:,a-int(patch_size/2):a+int(patch_size/2),b-int(patch_size/2):b+int(patch_size/2)]

        img_rows = []
        for min_id in range(0, 16, 2):
            img_row = np.hstack([i for i in x[min_id:min_id+2]])
            img_rows.append(img_row.transpose(2, 1, 0))
        img = np.vstack(img_rows)

        img = img[:,:,::-1] # BGR to RGB
        min, max = img.min(), img.max()
        sen_img = (img - min) / (max - min)

        _image = image.transpose(1,2,0) / 255.0
        new_img = np.vstack((_image, sen_img.transpose(1,0,2)))
        sen_img = cv2.resize(new_img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        sen_img = sen_img.transpose(2,0,1)
        ######################################

        image = torch.from_numpy(image).float()
        image = self.normalize_image(image) 

        sen_img = torch.from_numpy(sen_img).float()
        sen_img = self.normalize_image(sen_img) 

        return image, sen_img, img_id
    

def Datasetloader(df, cfg, centroid):   
    valid_dataset = NetDataset(df, cfg, "val", centroid)
    validloader = torch.utils.data.DataLoader(valid_dataset, 
                                              batch_size=cfg.batch_size, 
                                              shuffle=False, 
                                              num_workers=cfg.num_workers, 
                                              pin_memory=True,
                                             )

    return validloader


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        dropout=0,
    ):
        super().__init__()

        # Convole input embedding and upscaled embedding
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout_skip = nn.Dropout(p=dropout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            skip = self.dropout_skip(skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        use_batchnorm=True,
        dropout=0,
    ):
        super().__init__()

        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]

        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
    

def _check_reduction(reduction_factors):
    """
    Assume features are reduced by factor of 2 at each stage.
    For example, convnext start with stride=4 and does not satisfy this condition.
    """
    r_prev = 1
    for r in reduction_factors:
        if r / r_prev != 2:
            raise AssertionError('Reduction assumed to increase by 2: {}'.format(reduction_factors))
        r_prev = r

class Model(torch.nn.Module):
    # The U-Net model
    # See also TimmUniversalEncoder in segmentation_models_pytorch
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = timm.create_model(self.cfg.model_name, features_only=True, pretrained=cfg.pretrained)
        encoder_channels = self.encoder.feature_info.channels()
        _check_reduction(self.encoder.feature_info.reduction())
        decoder_channels = [ch * cfg.decoder_ch_coef for ch in (256, 128, 64, 32, 16)]
        assert len(encoder_channels) == len(decoder_channels)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=self.cfg.dropout,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=self.cfg.num_classes, activation=None, kernel_size=3,
        )
        
        # Test-time augmentation
        self.tta = TTA(cfg.num_tta) #cfg['tta']

        initialize_decoder(self.decoder)

    def forward(self, x):
        if self.cfg.tta == True:
            x = self.tta.stack(x)
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        y_pred = self.segmentation_head(decoder_output)
        if self.cfg.tta == True:
            y_pred = self.tta.average(y_pred)
        return y_pred


def test(df, cfg, centroid):
    validloader = Datasetloader(df, cfg, centroid)

    models = []
    for i in range(len(cfg.models_data)):
        cfg.model_name = cfg.models_data[i][0]
        cfg.decoder_ch_coef = cfg.models_data[i][2]
        model = Model(cfg).to(cfg.device).eval()
        model.load_state_dict(torch.load("models/" + cfg.models_data[i][1]))
        models.append(model)

    weight_sum = 0.0
    with torch.no_grad():
        for (x, x_sen, id) in tqdm(validloader):
            x, x_sen = x.to(cfg.device), x_sen.to(cfg.device)
            logits = []
            for i in range(len(models)):
                with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp):
                    if i < len(models)-1:
                        logit = models[i](x)
                    else: logit = models[i](x_sen)
                    logits.append(logit * cfg.models_data[i][3])
                    weight_sum += cfg.models_data[i][2]
            
            logits = sum(logits) / weight_sum
            preds = logits.argmax(dim=1) 
            preds = preds.cpu().numpy().astype('uint8')
            for i in range(len(preds)):
                output_file = Path(cfg.output_dir, id[i].split('/')[-1].replace('IMG', 'PRED'))
                Image.fromarray(preds[i]).save(output_file, compression='tiff_lzw')



def main():
    cfg = Cfg()
    df = pd.read_csv("test.csv")
    
    centroid = json.load(open("flair-2_centroids_sp_to_patch.json"))
    test(df, cfg, centroid)


if __name__ == "__main__":
    main()



