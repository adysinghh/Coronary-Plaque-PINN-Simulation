"""
Data Ingestion and Preprocessing module for the COCA dataset.
This module provides functions to recursively load DICOM file paths,
preprocess each image (normalize and resize), and a custom PyTorch Dataset.
"""

import os
import numpy as np
import pydicom
import cv2
import torch
from torch.utils.data import Dataset

def load_dicom_files(root_dir):
    """
    Recursively load DICOM file paths from the specified root directory.
    """
    dicom_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_paths.append(os.path.join(subdir, file))
    print(f"Found {len(dicom_paths)} DICOM files in {root_dir}")
    return dicom_paths

def preprocess_dicom(file_path, target_size=(512, 512)):
    """
    Read a DICOM file, extract pixel data, normalize and resize.
    
    Args:
        file_path (str): Path to the DICOM file.
        target_size (tuple): Desired output image size (width, height).
    
    Returns:
        np.ndarray: Preprocessed image as a 2D numpy array.
    """
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    # Normalize to [0, 1]
    img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-8)
    # Resize using bilinear interpolation
    img_resized = cv2.resize(img_norm, target_size, interpolation=cv2.INTER_LINEAR)
    return img_resized

class COCADataset(Dataset):
    """
    PyTorch Dataset for the COCA dataset.
    
    Each sample is a preprocessed DICOM image with an added channel dimension.
    """
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = preprocess_dicom(self.file_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        # Add channel dimension: shape (1, H, W)
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img, dtype=torch.float32)
