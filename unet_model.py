#!/usr/bin/env python
"""
U-Net Segmentation Model module.
Defines the U-Net architecture, a combined loss function (BCE + Dice Loss),
and a training routine for the segmentation model with tqdm progress bars.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class DoubleConv(nn.Module):
    """
    Double convolution block: two sequential conv->BN->ReLU layers.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    U-Net architecture with encoder, bottleneck, and decoder with skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # Decoder (Upsampling)
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)
        return self.final_conv(x)

def dice_loss(pred, target, smooth=1e-6):
    """
    Compute the Dice Loss for binary segmentation.
    """
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combined loss function: BCEWithLogitsLoss and Dice Loss.
    """
    def __init__(self, weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight = weight

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = dice_loss(pred, target)
        return self.weight * loss_bce + (1 - self.weight) * loss_dice

def train_unet(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """
    Train the U-Net model using the CombinedLoss.
    Shows progress with tqdm and saves the best model based on validation loss.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CombinedLoss(weight=0.5)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # Using tqdm for the training loop progress
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            # Here we assume self-supervision; replace 'batch' with ground truth mask if available.
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
    return model
