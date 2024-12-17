import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from tomato_dataset import TomatoDataset
from unet_model import UNet

def main():
    # Configuration
    DATA_ROOT = 'laboro_tomato'
    BATCH_SIZE = 24
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets and dataloaders
    train_dataset = TomatoDataset(DATA_ROOT, split='train')
    test_dataset = TomatoDataset(DATA_ROOT, split='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)

    # Create model, loss, and optimizer
    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output_{timestamp}'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}') as pbar:
            for images, masks, _ in pbar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))

if __name__ == '__main__':
    main()
