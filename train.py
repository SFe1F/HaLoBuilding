import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", default='config/HaLoBuilding.py')
    return parser.parse_args()

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if 'img' in batch:
            img = batch['img']
        elif 'image' in batch:
            img = batch['image']
        else:
            raise KeyError("数据批次字典中必须包含 'img' 或 'image' 键")

        if 'gt_semantic_seg' in batch:
            mask = batch['gt_semantic_seg']
        elif 'mask' in batch:
            mask = batch['mask']
        else:
            raise KeyError("数据批次字典中必须包含 'gt_semantic_seg' 或 'mask' 键")
        
        prediction = self.net(img)
        loss = self.loss(prediction, mask)
        
        self.log('train_step_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_losses.append(loss.item())
        
        pre_mask_probs = torch.softmax(prediction, dim=1)
        pre_mask = torch.argmax(pre_mask_probs, dim=1)
        
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
            
        return loss

    def validation_step(self, batch, batch_idx):
        if 'img' in batch:
            img = batch['img']
        elif 'image' in batch:
            img = batch['image']
        else:
            raise KeyError("数据批次字典中必须包含 'img' 或 'image' 键")

        if 'gt_semantic_seg' in batch:
            mask = batch['gt_semantic_seg']
        elif 'mask' in batch:
            mask = batch['mask']
        else:
            raise KeyError("数据批次字典中必须包含 'gt_semantic_seg' 或 'mask' 键")
        
        prediction = self.forward(img)
        loss_val = self.loss(prediction, mask)
        self.val_losses.append(loss_val.item())
        
        pre_mask_probs = torch.softmax(prediction, dim=1)
        pre_mask = torch.argmax(pre_mask_probs, dim=1)

        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
            
        self.log('val_step_loss', loss_val, on_step=True, on_epoch=False)


    def on_train_epoch_start(self):
        self.train_losses = []
        self.metrics_train.reset()

    def on_train_epoch_end(self):
        if not self.train_losses: return
        avg_train_loss = np.mean(self.train_losses)
        
        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())

        print(f"\nEpoch {self.current_epoch} 训练结果:")
        print(f"平均损失: {avg_train_loss:.4f} | mIoU: {mIoU:.4f} | F1: {F1:.4f} | OA: {OA:.4f}")

        iou_per_class = self.metrics_train.Intersection_over_Union()
        print("各类IoU:")
        for class_name, iou in zip(self.config.classes, iou_per_class):
            print(f"- {class_name}: {iou:.4f}")
            
        log_dict = {
            'train_loss_epoch': avg_train_loss,
            'train_mIoU_epoch': mIoU,
        }
        self.log_dict(log_dict, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_losses = []
        self.metrics_val.reset()

    def on_validation_epoch_end(self):
        if not self.val_losses: return
        avg_val_loss = np.mean(self.val_losses)
        
        iou_per_class = self.metrics_val.Intersection_over_Union()
        mIoU = np.nanmean(iou_per_class)
        F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())
        
        building_iou = iou_per_class[1]
        
        print("\n" + "="*50)
        print(f"Epoch {self.current_epoch} 验证结果:")
        print(f"平均损失: {avg_val_loss:.4f} | mIoU: {mIoU:.4f} | F1: {F1:.4f} | OA: {OA:.4f}")
        print(f"建筑物的IoU: {building_iou:.4f}")
        
        print("各类IoU:")
        for class_name, iou in zip(self.config.classes, iou_per_class):
            print(f"- {class_name}: {iou:.4f}")
            
        log_dict = {
            'val_loss_epoch': avg_val_loss,
            'val_mIoU_epoch': mIoU,
            'val_building_IoU': building_iou,
        }
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k, 
        monitor=config.monitor,
        save_last=config.save_last, 
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name
    )
    
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)  
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(
            config.pretrained_ckpt_path, 
            config=config, 
            strict=False 
        )

    trainer = pl.Trainer(
        devices=config.gpus, 
        max_epochs=config.max_epoch, 
        accelerator='auto',
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        strategy='auto',
        logger=logger
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)

if __name__ == "__main__":
    main()