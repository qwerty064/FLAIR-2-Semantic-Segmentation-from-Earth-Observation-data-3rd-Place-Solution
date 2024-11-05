from pathlib import Path
import json
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
import cv2
from tta import TTA
from PIL import Image
from pathlib import Path


class Cfg:
    model_name              = "maxvit_rmlp_pico_rw_256.sw_in1k" 
    decoder_ch_coef         = 1
    tta                     = False
    num_tta                 = 4 # 4, 8
    pretrained              = False
    batch_size              = 16
    img_size                = 512
    num_classes             = 13 
    dropout                 = 0.0

    device                  = "cuda"
    use_amp                 = True 
    compile                 = False
    exp_name                = "pico_sentinel"   
    num_workers             = 4

    model_path              = "models/pico_sentinel.pt"
    decoder_ch_coef         = 1
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
            image = np.array(img.read()[:3]) #[:3]
        image = image.transpose(1, 2, 0)

        parts = img_path.split('/')
        sen_path = "flair_2_sen_test/"
        sen_img_path = sen_path + parts[-4] +'/'+ parts[-3] + "/sen/SEN2_sp_" + parts[-4]+'-'+parts[-3]+"_data.npy"

        x = np.load(sen_img_path)
        indices = np.random.randint(x.shape[0], size=16) # take random 16
        x = x[indices]

        patch_size = 64 # 40-112
        a, b = self.centroids[parts[-1]]
        x = x[:,:3,::] #BGR
        x = x[:,:,a-int(patch_size/2):a+int(patch_size/2),b-int(patch_size/2):b+int(patch_size/2)]

        img_rows = []
        for min_id in range(0, 16, 2):
            img_row = np.hstack([i for i in x[min_id:min_id+2]])
            img_rows.append(img_row.transpose(2, 1, 0))
        img = np.vstack(img_rows)

        img = img[:,:,::-1] #RGB
        min, max = img.min(), img.max()
        sen_img = (img - min) / (max - min)

        image = image / 255.0
        new_img = np.vstack((image, sen_img.transpose(1,0,2)))
        new_img = cv2.resize(new_img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        image = new_img.transpose(2,0,1)
        image = torch.from_numpy(image).float()
        image = self.normalize_image(image) 

        return image, img_id
    

def Datasetloader(df, cfg, centroid):   
    valid_dataset = NetDataset(df, cfg, "val", centroid)
    validloader = torch.utils.data.DataLoader(valid_dataset, 
                                              batch_size=cfg.batch_size, 
                                              shuffle=False, 
                                              num_workers=cfg.num_workers, 
                                              pin_memory=True,
                                              persistent_workers=True)

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
        self.tta = TTA(cfg.num_tta)

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
    model = Model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_path))
    if cfg.compile == True:
        model = torch.compile(model)
    model.eval()

    with torch.no_grad():
        for (x,id) in tqdm(validloader):
            x = x.to(cfg.device)
            with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp):
                logits = model(x)
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



