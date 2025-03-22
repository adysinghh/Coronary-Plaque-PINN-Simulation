import os
from torch.utils.data import DataLoader
import torch

from data_preprocessing import load_dicom_files, COCADataset
from unet_model import UNet, train_unet
from pinn_simulation import train_pinn_with_geometry
from dashboard import launch_dashboard

TEST_MODE = True
FORCE_TRAINING_PINN = True

def main():
    root_dir = "./dataset/cocacoronarycalciumandchestcts-2/Gated_release_final"
    dicom_files = load_dicom_files(root_dir)
    if TEST_MODE:
        dicom_files = dicom_files[:50]
        print("TEST_MODE enabled: using a small sample of the dataset.")
    total_files = len(dicom_files)
    train_files = dicom_files[:int(0.8 * total_files)]
    val_files = dicom_files[int(0.8 * total_files):int(0.9 * total_files)]
    test_files = dicom_files[int(0.9 * total_files):]

    train_dataset = COCADataset(train_files)
    val_dataset = COCADataset(val_files)
    test_dataset = COCADataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # U-Net
    unet_model = UNet(in_channels=1, out_channels=1)
    unet_checkpoint = "models/best_unet_model.pth"
    if os.path.exists(unet_checkpoint):
        print("Loading pre-trained U-Net segmentation model...")
        unet_model.load_state_dict(torch.load(unet_checkpoint, map_location=device))
    else:
        print("Training U-Net segmentation model...")
        unet_model = train_unet(unet_model, train_loader, val_loader, num_epochs=10 if TEST_MODE else 50, device=device)
        torch.save(unet_model.state_dict(), unet_checkpoint)
    unet_model.eval()

    # PINN
    pinn_checkpoint = "models/pinn_model_checkpoint"
    try:
        if not FORCE_TRAINING_PINN and os.path.exists(pinn_checkpoint):
            print("Restoring pre-trained PINN model for bounding box geometry...")
            # Minimal training for structure
            from pinn_simulation import dde
            pinn_model, pinn_predict = train_pinn_with_geometry([[0, 0], [1, 1]], num_epochs=1)
            pinn_model.restore(pinn_checkpoint)
        else:
            if TEST_MODE:
                print("TEST_MODE: training bounding box PINN for fewer epochs...")
                pinn_model, pinn_predict = train_pinn_with_geometry([[0, 0], [1, 1]], num_epochs=50)
            else:
                pinn_model, pinn_predict = train_pinn_with_geometry([[0, 0], [1, 1]], num_epochs=5000)
            pinn_model.save(pinn_checkpoint)
    except Exception as e:
        print("Error loading PINN model:", e)
        print("Training bounding box PINN from scratch...")
        if TEST_MODE:
            pinn_model, pinn_predict = train_pinn_with_geometry([[0, 0], [1, 1]], num_epochs=50)
        else:
            pinn_model, pinn_predict = train_pinn_with_geometry([[0, 0], [1, 1]], num_epochs=5000)
        pinn_model.save(pinn_checkpoint)

    # Dashboard
    launch_dashboard(unet_model, device, test_loader, pinn_predict)

if __name__ == "__main__":
    main()
