import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base.initialization import initialize_decoder
import torchvision.transforms
from transformers import get_cosine_schedule_with_warmup
import rasterio
from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp


class Cfg:
    pretrained              = True
    num_epochs              = 10
    lr                      = 1e-4
    img_size                = 512
    num_classes             = 13 
    optimizer               = "AdamW"
    loss_fn                 = "CrossEntropyLoss" # CrossEntropyLoss, FocalLoss
    scheduler               = "cosine_schedule_with_warmup"
    adamW_eps               = 1e-08 # 1e-4
    warmup_steps_ratio      = 0.1
    dropout                 = 0.0
    n_data                  = -1
    n_splits                = 1 # 1 for full data train, 5, 10

    device                  = "cuda"
    use_amp                 = True 
    compile                 = False
    save_model              = True
    save_epochs             = [10] 
    num_workers             = 4

    # change these parameters for every different model
    ######################################## 
    exp_name                = "id0" 
    model_name              = "maxvit_tiny_tf_512.in1k" 
    decoder_ch_coef         = 2 
    batch_size              = 8 
    grad_clip               = 2.0 # None
    iters_to_accumulate     = 1
    seed                    = 1
    # #######################################
    # exp_name                = "id1" 
    # model_name              = "maxvit_tiny_tf_512.in1k" 
    # decoder_ch_coef         = 4 
    # batch_size              = 8 
    # grad_clip               = 2.0
    # iters_to_accumulate     = 1 
    # seed                    = 2
    # #####################################
    # exp_name                = "id3" 
    # model_name              = "maxvit_tiny_tf_512.in1k" 
    # decoder_ch_coef         = 4 
    # batch_size              = 8  
    # grad_clip               = 2.0
    # iters_to_accumulate     = 1 
    # seed                    = 4
    # ######################################
    # exp_name                = "id2" 
    # model_name              = "maxvit_base_tf_512.in21k_ft_in1k" 
    # lr                      = 4e-5
    # decoder_ch_coef         = 4 
    # batch_size              = 4
    # adamW_eps               = 1e-4
    # grad_clip               = 0.7
    # iters_to_accumulate     = 2
    # seed                    = 3
    # ######################################
    # # This model loss goes NaN after 4 epochs, so it might hard to reproduce 
    # exp_name                = "id7" 
    # model_name              = "maxvit_base_tf_512.in21k_ft_in1k" 
    # lr                      = 1e-4
    # decoder_ch_coef         = 1
    # batch_size              = 4
    # adamW_eps               = 1e-08  
    # grad_clip               = 0.7
    # iters_to_accumulate     = 2
    # seed                    = 0
    # save_epochs             = [3,4,5,6,7,8,9,10] 


class NetDataset(Dataset):
    def __init__(self, df, cfg, transform):
        super().__init__()
        self.cfg = cfg
        self.df = np.array(df)
        self.transform = transform
        self.normalize_image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.train_transforms = A.Compose(
            [
                A.RandomRotate90(p=1), 
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(rotate_limit=30, scale_limit=0.2, p=0.75),  
                A.RandomResizedCrop(
                    self.cfg.img_size,
                    self.cfg.img_size,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1111111111111),
                    p=0.5,
                ),
                ToTensorV2(transpose_mask=True), # HWC to CHW
            ]
        )
        
        self.val_transforms = A.Compose(
            [   
                ToTensorV2(transpose_mask=True),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        data = self.df[i]
        img_path, label_path = data[0], data[1]

        with rasterio.open(img_path) as img:
            new_img = np.array(img.read()[:3]) 
        image = new_img.transpose(1, 2, 0)

        with rasterio.open(label_path) as labell:
            label = np.array(labell.read())
        label[label > self.cfg.num_classes] = self.cfg.num_classes 
        label = label-1
        label = label.squeeze()

        if self.transform == "train":
            transformed = self.train_transforms(image=image, mask=label)
        else: transformed = self.val_transforms(image=image, mask=label)
        image, label = transformed["image"], transformed["mask"]

        image = image.float()
        image = self.normalize_image(image)

        return image, label
    

def Datasetloader(df, cfg, fold):
    validloader = None
    shuffle = True
    if cfg.n_data != -1:
        df = df.iloc[: cfg.n_data]  
        # df = df.sample(frac=0.5).reset_index(drop=True)
    if cfg.n_splits != 1:
        skf = GroupKFold(n_splits=cfg.n_splits)
        train_idx, val_idx = list(skf.split(df, groups=df.area))[fold]
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()        
        valid_dataset = NetDataset(val_df, cfg, "val")
        validloader = DataLoader(valid_dataset, 
                                 batch_size=cfg.batch_size, 
                                 shuffle=False, 
                                 num_workers=cfg.num_workers, 
                                 pin_memory=True)  
        
    elif cfg.n_splits == 1:
        train_df = df

    train_dataset = NetDataset(train_df, cfg, "train")
    trainloader = DataLoader(train_dataset, 
                             batch_size=cfg.batch_size, 
                             shuffle=shuffle,
                             num_workers=cfg.num_workers,
                             pin_memory=True,
                             drop_last=True)

    return trainloader, validloader


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

class Model(nn.Module):
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

        initialize_decoder(self.decoder)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        y_pred= self.segmentation_head(decoder_output)
        return y_pred 

    # def set_grad_checkpointing(self, enable: bool = True):
    #     self.backbone.encoder.model.set_grad_checkpointing(enable)


def train_one_epoch(model, trainloader, cfg, scaler, optimizer, scheduler, epoch, loss_fn, metric):
    epoch_loss = 0.0
    model.train()
    for i, (x,y) in enumerate(tqdm(trainloader)): 
        x, y = x.to(cfg.device), y.long().to(cfg.device) 
        with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        loss = loss / cfg.iters_to_accumulate

        scaler.scale(loss).backward() # loss.backward()

        if (i + 1) % cfg.iters_to_accumulate == 0:
            if cfg.grad_clip != None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)

            scaler.step(optimizer) # optimizer.step()
            scaler.update()
            scheduler.step() 
            optimizer.zero_grad()

        preds = logits.argmax(dim=1) 
        metric(preds=preds, target=y)
        epoch_loss += loss.item() * x.size(0)

    if (epoch+1) in cfg.save_epochs and cfg.save_model:
        torch.save(model.state_dict(), f"models/{cfg.model_name}_{cfg.exp_name}_epoch{epoch+1}.pt") 
    
    return epoch_loss, metric


def eval_model(model, validloader, cfg, loss_fn, metric, metric2):
    epoch_loss = 0.0
    model.eval()
    preds = []
    with torch.no_grad():
        for (x,y) in tqdm(validloader):
            x, y = x.to(cfg.device), y.long().to(cfg.device) 
            with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            preds = logits.argmax(dim=1) 
            metric(preds=preds, target=y)
            metric2(preds=preds, target=y)
            epoch_loss += loss.item() * x.size(0)
    return epoch_loss, metric, metric2


def train(df, cfg):
    for fold in range(cfg.n_splits):
        trainloader, validloader = Datasetloader(df, cfg, fold)
        model = Model(cfg).to(cfg.device)
        if cfg.compile == True:
            model = torch.compile(model)
        
        print("-" * 40)
        print(f"Fold {fold+1}/{cfg.n_splits}")

        if cfg.loss_fn == "CrossEntropyLoss":
            loss_fn = nn.CrossEntropyLoss().to(cfg.device)
        elif cfg.loss_fn == "FocalLoss":
            loss_fn = smp.losses.FocalLoss("multiclass", alpha=0.25, gamma=2.0).to(cfg.device)

        metric = JaccardIndex(task='multiclass',num_classes=cfg.num_classes,average='macro').to(cfg.device)
        metric2 = JaccardIndex(task='multiclass',num_classes=cfg.num_classes,average='micro').to(cfg.device)

        if cfg.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.adamW_eps)
        elif cfg.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        if cfg.scheduler == "cosine_schedule_with_warmup":
            total_steps = len(trainloader)*cfg.num_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=round(total_steps*cfg.warmup_steps_ratio), 
                num_training_steps=total_steps
            )
            
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp) # scaler = None

        for epoch in range(cfg.num_epochs):
            print("-" * 10)
            print(f"Epoch {epoch+1}/{cfg.num_epochs}")

            # Training
            running_loss, metric = train_one_epoch(model, trainloader, cfg, scaler, optimizer, scheduler, epoch, loss_fn, metric)  
            epoch_loss = running_loss / len(trainloader.dataset)
            score = metric.compute()
            metric.reset()
            print(f"Train Loss: {epoch_loss:.4f} | Score: {score:.4f}")

            # Validation
            if validloader != None:
                running_loss, metric, metric2 = eval_model(model, validloader, cfg, loss_fn, metric, metric2)
                epoch_loss = running_loss / len(validloader.dataset)
                score = metric.compute()
                score2 = metric2.compute()
                metric.reset()
                metric2.reset()
                print(f"Valid Loss: {epoch_loss:.4f} | Score: {score:.4f} | Score2: {score2:.4f}")


def main():
    cfg = Cfg()

    seed = cfg.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False

    if cfg.use_amp == False:
        torch.set_float32_matmul_precision('high')

    df = pd.read_csv("train.csv")
    df = df.sample(frac=1)

    train(df, cfg)


if __name__ == "__main__":
    main()

