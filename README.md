# Tomato Segmentation with U-Net

A deep learning project for semantic segmentation of tomato images using a U-Net architecture. This implementation is designed to run on SLURM-based computing clusters and uses PyTorch for model training and inference.

## Project Overview

This project implements a U-Net model for precise tomato segmentation in images. The model is trained on the Laboro Tomato dataset and includes features such as:
- Custom dataset handling with automatic image orientation correction
- Data augmentation for improved model robustness
- SLURM-based training and testing scripts for HPC environments
- Comprehensive evaluation metrics including IoU (Intersection over Union)

## Repository Setup

Clone the repository:
```bash
git clone https://github.com/jan3zk/tomato_segmentation.git
cd tomato_segmentation/
```

## Dataset Structure
The dataset can be downloaded from [this link](https://unilj-my.sharepoint.com/:u:/g/personal/mivanovska_fe1_uni-lj_si/EUrSvUOGv6dBnlaJyFqVx5MB6SEV2NXf11uEgnwvX3UVFQ?e=SJcZls).

The dataset follows the structure described below:

```
laboro_tomato/
├── train/
│   ├── images/tomato/*.png    # Training images
│   ├── masks/tomato/*.png     # Training masks
│   └── gt/tomato/*.png        # Ground truth masks
├── test/
│   ├── images/tomato/*.png    # Testing images
│   ├── masks/tomato/*.png     # Testing masks
│   └── gt/tomato/*.png        # Ground truth masks
└── annotations.json            # Bounding box and ripeness labels
```

## Training

Submit the training job:
```bash
sbatch train.sh
```

The training script:
- Loads the dataset from the `laboro_tomato` directory.
- Trains the U-Net model.
- Saves the best model checkpoint in the `output_<timestamp>/checkpoints` directory.

## Testing

Submit the testing job with the model checkpoint path:
```bash
sbatch test.sh output_<timestamp>/checkpoints/best_model.pt
```

The testing script:
- Loads the trained model.
- Evaluates on the test dataset.
- Saves predictions and evaluation results in a `test_results_<timestamp>` directory.

## Model Architecture
The U-Net model includes:
- **Encoder:** Extracts image features using convolutional layers.
- **Decoder:** Reconstructs the segmentation mask with skip connections.
