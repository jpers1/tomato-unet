import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2   

class TomatoDataset(Dataset):
    def __init__(self, root, split='train'):
        self.split = split
        self.images_dir = os.path.join(root, split, 'images', 'tomato')
        if os.path.exists(os.path.join(root, split, 'mask')):
            self.masks_dir = os.path.join(root, split, 'mask', 'tomato')
        else:
            self.masks_dir = os.path.join(root, split, 'masks', 'tomato')
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        print(f"Found {len(self.image_files)} images in {split} set")

        # Define augmentations for training data
        if split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.GaussNoise(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Only normalize and convert to tensor for test data
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Check orientation and rotate if needed
        if image.size[1] > image.size[0]:  # if height > width (portrait)
            image = image.rotate(90, expand=True)
            mask = mask.rotate(90, expand=True)
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask) > 0
        mask = mask.astype(np.float32)
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)  # Add channel dimension
        
        return image, mask, img_name

    def __len__(self):
        return len(self.image_files)