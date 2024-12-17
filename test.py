# test.py
import os
import torch
from train import TomatoDataset, UNet
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def evaluate_model(model, data_loader, device, output_dir=None):
    model.eval()
    total_iou = 0
    num_samples = 0
    
    if output_dir:
        predictions_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, masks, filenames in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_masks = (torch.sigmoid(outputs) > 0.5)
            
            for pred, target, filename in zip(pred_masks, masks, filenames):
                iou = calculate_iou(pred, target)
                total_iou += iou
                num_samples += 1
                
                if output_dir:
                    pred_np = pred.squeeze().cpu().numpy()
                    plt.imsave(os.path.join(predictions_dir, f'pred_{filename}'), 
                             pred_np, cmap='gray')
    
    mean_iou = total_iou / num_samples
    return mean_iou

def calculate_iou(pred, target):
    intersection = torch.logical_and(pred, target).sum().float()
    union = torch.logical_or(pred, target).sum().float()
    iou = intersection / (union + 1e-7)
    return iou.item()

def main():
    parser = argparse.ArgumentParser(description='Test UNet on tomato dataset')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='laboro_tomato',
                      help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for testing')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = TomatoDataset(args.data_root, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # Create model and load weights
    model = UNet().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print available checkpoint information
    print("\nCheckpoint information:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], (int, float)) and key != 'model_state_dict':
            if isinstance(checkpoint[key], float):
                print(f"{key}: {checkpoint[key]:.4f}")
            else:
                print(f"{key}: {checkpoint[key]}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'test_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_iou = evaluate_model(model, test_loader, device, output_dir)
    print(f'Test Set IoU: {test_iou:.4f}')

    # Save results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f'Model path: {args.model_path}\n')
        f.write(f'Test Set IoU: {test_iou:.4f}\n')
        f.write('\nCheckpoint information:\n')
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], (int, float)) and key != 'model_state_dict':
                if isinstance(checkpoint[key], float):
                    f.write(f"{key}: {checkpoint[key]:.4f}\n")
                else:
                    f.write(f"{key}: {checkpoint[key]}\n")

if __name__ == '__main__':
    main()