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

## Dataset
Download the dataset using [gdown](https://github.com/wkentaro/gdown):

```bash
gdown --id 1K5Zo47YIddzO3OnLgOJJbQig8TC_uSjb
```

If `gdown` is not installed you can add it with:

```bash
pip install --user gdown
# or using conda
conda install -c conda-forge gdown
```

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

## Running in batch mode on HPC

Submit the training job:
```bash
sbatch train.sh
```

The training script:
- Loads the dataset from the `laboro_tomato` directory.
- Trains the U-Net model.
- Saves the best model checkpoint in the `output_<timestamp>/checkpoints` directory.

Submit the testing job with the model checkpoint path:
```bash
sbatch test.sh output_<timestamp>/checkpoints/best_model.pt
```

The testing script:
- Loads the trained model.
- Evaluates on the test dataset.
- Saves predictions and evaluation results in a `test_results_<timestamp>` directory.

## Running in interactive mode on HPC

For interactive sessions we recommend installing [uhpc-tools](https://github.com/jpers1/uhpc-tools). After installation you can request an interactive GPU session and run `python train.py` or `python test.py` directly.

## Model Architecture
The U-Net model includes:
- **Encoder:** Extracts image features using convolutional layers.
- **Decoder:** Reconstructs the segmentation mask with skip connections.
