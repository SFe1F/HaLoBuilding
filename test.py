import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from thop import profile
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
from catalyst import utils
try:
    from geoseg.datasets.HaLo_H import CLASSES, PALETTE
except ImportError:
    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    
    color_map = {
        0: [0, 0, 0],       
        1: [255, 255, 255], 
    }

    for class_id, color in color_map.items():
        mask_rgb[mask == class_id] = color
        
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb, output_path) = inp
    if rgb:
        mask_name_png = os.path.join(output_path, mask_id + '.png')
        mask_rgb = label2rgb(mask)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_png, mask_bgr)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = os.path.join(output_path, mask_id + '.png')
        cv2.imwrite(mask_name_png, mask_png)


def rgb_to_class_indices(rgb_image, color_map):
    if len(rgb_image.shape) == 2:
        rgb_image = np.stack([rgb_image] * 3, axis=-1)
    
    h, w, _ = rgb_image.shape
    class_indices = np.zeros((h, w), dtype=np.uint8)
    
    for class_idx, color in enumerate(color_map):
        color_arr = np.array(color)
        class_indices[np.all(rgb_image == color_arr, axis=-1)] = class_idx
    
    return class_indices

def calculate_global_metrics(label_images, predict_images, num_classes):
    label_flat = np.concatenate([label.flatten() for label in label_images])
    predict_flat = np.concatenate([predict.flatten() for predict in predict_images])
    mask = label_flat < num_classes
    label_flat = label_flat[mask]
    predict_flat = predict_flat[mask]
    
    cm = confusion_matrix(label_flat, predict_flat, labels=range(num_classes))
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    miou = np.mean(iou)

    return iou, f1, precision, miou


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, default='config/loveda/unetformer_lwganet_l2.py', help="Path to config")
    arg("-o", "--output_path", type=Path, default='/data/sff/LWGANet-main/segmentation/result_Ablation/LSK_fre/foggy3.0', help="Path where to save resulting masks.")
    arg("-t", "--tta", help="Test time augmentation.", default ='d4', choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    arg("--test_eval", help="whether eval test set", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda:0")
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model_path = os.path.join(config.weights_path, f"{config.test_weights_name}.ckpt")
    print(f"正在加载模型: {model_path}")
    model = Supervision_Train.load_from_checkpoint(model_path, config=config, map_location=device, strict=False)
    # model.cuda()
    model = model.to(device)
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    is_eval_mode = args.val or args.test_eval
    if is_eval_mode:
        if args.val:
            test_dataset = config.val_dataset
        elif args.test_eval:
            from geoseg.datasets.HaLo_H import loveda_test_with_mask_dataset
            test_dataset = loveda_test_with_mask_dataset
    else:
        test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        results = []
        all_predictions = []
        all_true_masks = []
        
        for input in tqdm(test_loader):
            raw_predictions = model(input['img'].cuda())
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            image_ids = input["img_id"]
            
            for i in range(predictions.shape[0]):
                mask_predict = predictions[i].cpu().numpy()
                all_predictions.append(mask_predict)
                
                if is_eval_mode:
                    mask_true = input['gt_semantic_seg'][i].cpu().numpy()
                    all_true_masks.append(mask_true)

                results.append((mask_predict, image_ids[i], args.rgb, str(args.output_path)))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    print('images writing spends: {:.2f} s'.format(t1 - t0))

    if is_eval_mode:
        print("\nCalculating metrics...")
        iou_per_class, f1_per_class, precision_per_class, miou = calculate_global_metrics(all_true_masks, all_predictions, len(CLASSES))
        
        print("Class-wise Metrics:")
        for i in range(len(CLASSES)):
            print(f"Class {i} ({CLASSES[i]}):")
            print(f"  IoU: {iou_per_class[i]:.4f}")
            print(f"  F1: {f1_per_class[i]:.4f}")
            print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"mIoU: {miou:.4f}")


if __name__ == "__main__":
    main()