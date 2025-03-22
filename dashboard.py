#!/usr/bin/env python
"""
Streamlit Dashboard module.
Displays the U-Net segmentation overlay alongside the original CT image,
followed by the PINN simulation output.
Includes explanatory details for each section and dataset information.
"""

import torch
import numpy as np
import streamlit as st
import cv2

def create_segmentation_overlay(original_img, seg_mask):
    """
    Creates an overlay image by blending the original grayscale image with a red-colored segmentation mask.
    """
    # Assume original_img is in [0,1] and scale to 0-255 for visualization.
    original_uint8 = (original_img * 255).astype(np.uint8)
    original_color = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2BGR)
    
    # Create a red mask for the segmentation (BGR: (0,0,255))
    red_mask = np.zeros_like(original_color)
    red_mask[:, :, 2] = seg_mask  # Use seg_mask (assumed to be 0 or 255) as red channel
    
    # Blend images: 70% original, 30% red overlay
    overlay = cv2.addWeighted(original_color, 0.7, red_mask, 0.3, 0)
    return overlay

def launch_dashboard(unet_model, device, test_loader, pinn_predict):
    st.set_page_config(page_title="Coronary Plaque & Physics Simulation Dashboard", layout="wide")
    st.title("Coronary Plaque Segmentation & Integrated Physics-Informed Simulation")
    st.markdown(
        """
        **Overview:**  
        - **Left:** Segmentation overlay from U-Net on a CT image (calcified plaque highlighted in red).  
        - **Right:** The original unsegmented CT image.  
        - **Below:** The PINN simulation output (a normalized displacement field) computed on the domain defined by the segmentation bounding box.
        """
    )

    # --- Section 1: Segmentation & Original Image ---
    st.header("1. CT Image & Segmentation Overlay")
    unet_model.to(device)
    unet_model.eval()

    # Get one sample image from the test set
    sample_tensor = next(iter(test_loader))[0].to(device)  # shape (1, H, W)
    with torch.no_grad():
        seg_output = unet_model(sample_tensor.unsqueeze(0))
        seg_prob = torch.sigmoid(seg_output)[0, 0].cpu().numpy()
    
    # The original image is already normalized; retrieve it from sample_tensor
    original_img = sample_tensor.cpu().numpy().squeeze()  # shape (H, W)
    seg_mask = (seg_prob > 0.8).astype(np.uint8) * 255

    # Create segmentation overlay
    overlay_img = create_segmentation_overlay(original_img, seg_mask)

    # Resize images for display (e.g., 256x256)
    display_size = (256, 256)
    original_disp = cv2.resize((original_img*255).astype(np.uint8), display_size, interpolation=cv2.INTER_LINEAR)
    overlay_disp = cv2.resize(overlay_img, display_size, interpolation=cv2.INTER_LINEAR)

    # Use two columns: Left for overlay, right for original image
    col1, col2 = st.columns(2)
    with col1:
        st.image(overlay_disp, caption="Segmentation Overlay", use_container_width=True)
    with col2:
        st.image(original_disp, caption="Original CT Image", use_container_width=True)
    st.markdown(
        """
        **Interpretation:**  
        The left image shows the CT scan with calcified plaque segmented and overlaid in red by the U-Net model.  
        The right image displays the original CT scan without any segmentation.  
        This comparison helps verify the quality of the segmentation.
        """
    )

    # --- Section 2: PINN Simulation Output ---
    st.header("2. PINN Simulation Output (Displacement Field)")
    resolution = 100
    x_vals = np.linspace(0, 1, resolution)
    y_vals = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    xy = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    u_pred = pinn_predict(xy)
    u_field = u_pred[:, 0].reshape(resolution, resolution)  # x-displacement field

    # Normalize and clip u_field to [0,1] for display
    min_val, max_val = u_field.min(), u_field.max()
    if abs(max_val - min_val) < 1e-12:
        st.write("PINN output is nearly constant; unable to display meaningful displacement field.")
        u_field_norm = u_field
    else:
        u_field_norm = (u_field - min_val) / (max_val - min_val + 1e-8)
        u_field_norm = np.clip(u_field_norm, 0.0, 1.0)
    
    st.image(u_field_norm, caption="Normalized x-Displacement Field from PINN Simulation", use_container_width=True)
    st.markdown(
        """
        **Interpretation:**  
        The simulation output represents the predicted x-displacement field from the PINN.  
        In our simplified model, the left boundary (black) is fixed (zero displacement) and the right boundary (white) shows a prescribed displacement.  
        This gradient reflects the solution of a 2D linear elasticity PDE on a domain derived from the segmentation.
        """
    )

    # --- Section 3: Dataset Information ---
    st.header("3. Dataset Information")
    st.markdown(
        """
        **COCA - Coronary Calcium and Chest CTs Dataset**  
        - **Source:** Stanford AIMI (COCA dataset)  
        - **Data Type:** Gated CT scans with detailed coronary images  
        - **Content:** Over 40,000 DICOM files capturing coronary arteries with calcified plaques  
        - **Usage:** Utilized for training a U-Net segmentation model to identify calcified regions.  
          The segmentation output informs the domain for a physics-informed simulation (PINN) that approximates stent expansion behavior.  
        - **Note:** For demonstration, a subset is used when TEST_MODE is enabled.
        """
    )
