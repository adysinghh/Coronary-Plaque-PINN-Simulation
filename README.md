! ![Flow Diagram](https://github.com/adysinghh/Coronary-Plaque-PINN-Simulation/blob/main/CoronaryPlaque-PINN-Simulation.png)

# Coronary-Plaque-PINN-Simulation
Integrated pipeline for coronary plaque segmentation using U-Net and simplified physics-informed simulation (PINN) over bounding-box–based domains extracted from CT scans. This is project is purely a Sample Implementation, To use this in real application you need several modification such as available ground truth to train the UNET, and useing the bounded box from the Image rather than the assumed [0,0] and  [1,1] bounded rectangular box.

![Segmentation + PINN Output](https://github.com/adysinghh/Coronary-Plaque-PINN-Simulation/raw/main/Image.png)

## Table of Contents

1. [What Is This Project About?](#what-is-this-project-about)  
2. [How Does It Work?](#how-does-it-work)  
3. [Main Parts of the Pipeline](#main-parts-of-the-pipeline)  
4. [Why Use a PINN?](#why-use-a-pinn)  
5. [Step-by-Step Guide](#step-by-step-guide)  
6. [Technical Terms in Simple Words](#technical-terms-in-simple-words)  
---

## 1. What Is This Project About?

Imagine you have **x-ray–like pictures** (CT scans) of a person’s heart arteries. Some of these arteries have **calcified plaque**—like tiny rock-like spots that can make the arteries stiff. We want to:

1. **Find** where that plaque is in the pictures.  
2. **Predict** how it might deform or change shape when a stent is placed.

To do this, we use **two** main tools:

- A **U‑Net**: A deep learning model that looks at a CT scan and colors in the spots of calcified plaque.  
- A **PINN**: A special neural network that obeys the laws of **physics** (like how materials stretch or bend).

---

## 2. How Does It Work?

1. **U‑Net** sees the **CT image** and marks where the calcified plaque is.  
2. It figures out a simple **bounding box** around that plaque (basically a rectangle that covers all the calcified pixels).  
3. The **PINN** uses that rectangle as a little “world” to solve a **2D elasticity** problem—like a mini simulation that says: “If one side is pinned and the other side is pushed, how does it stretch?”  
4. The **Streamlit** dashboard shows you both:
   - **Where the plaque is** on the original CT image.  
   - **How** the bounding box domain “stretches” in the simulation.

---

## 3. Main Parts of the Pipeline

1. **Data Preprocessing:**  
   - We load DICOM files (these are special medical image files) and turn them into Numpy arrays.  
   - We resize and normalize them so the U‑Net can process them consistently.

2. **U‑Net Segmentation:**  
   - The U‑Net is trained to exactly produce the same image, Since we dont have Ground truth available so we used Ground truth same as the input image, This is an demo implementation in real case we would require corresponding segmented image
   - It outputs the same Image.
   - 2.1. **Interpretation of the Final “Segmented” Mask**
        - Because the network is essentially reproducing the input CT image, the brightest areas in the U‑Net’s output will likely be the naturally bright spots (like bone or intense calcifications).
        - When we threshold at 0.8, we get a mask that might highlight some of those bright regions.
> [!NOTE]  
> This does not mean the network is truly learning to distinguish plaque from non-plaque. It’s just replicating the input and then we’re forcibly picking out high-intensity pixels.

### 3. **PINN Simulation:**

- **Domain Setup:**  
  - In our current demo, we use a hardcoded bounding box with coordinates [0, 0] and [1, 1] (a unit square).  
  - *In our case*, this unit square represents the simulation domain. In a real application, we would extract the actual bounding box from the segmented calcified plaque area and then normalize it to a standard domain.
  
- **The PDE Used – 2D Linear Elasticity:**  
  - We use a 2D elasticity PDE to model how a material (or in our case, the plaque region) deforms under applied forces.  
  - **Hooke’s Law** is built into this model—it relates the **strain** (how much the material deforms) to the **stress** (the internal forces resisting deformation) using material constants (Lamé constants).  
    
- **Boundary Displacement:**  
  - The simulation sets boundary conditions on the unit square:  
    - The **left boundary** (x = 0) is fixed (no displacement).  
    - The **right boundary** (x = 1) is given a prescribed displacement (for example, 0.1 in the x-direction).  
  - These conditions mimic the effect of, say, a stent expanding an artery—one side remains fixed while the other is pushed, causing the material to deform gradually.

- **Training the PINN:**  
  - The PINN is trained by minimizing a loss function that measures two things:  
    1. **PDE Residual Loss:** How well the network’s predicted displacement field satisfies the 2D elasticity PDE (i.e., the model “learns” the correct physics).  
    2. **Boundary Loss:** How well the network’s predictions match the specified boundary conditions (e.g., zero displacement on the left, 0.1 on the right).  
  - Optimizers like **Adam** (for initial training) and **L-BFGS** (for refinement) are used to adjust the network’s weights so that the overall loss is minimized.

- **Output – The Displacement Field:**  
  - After training, the PINN outputs a displacement field over the unit square.  
  - This output appears as a smooth gradient image:  
    - The left side (black) represents zero displacement.  
    - As you move to the right, the displacement gradually increases (tending toward white after normalization).  
  - This gradient illustrates how each point in the domain would move under the given force conditions.

4. **Dashboard Visualization:**  
   - A user-friendly **Streamlit** app that displays:
     - The original image + the U‑Net’s overlay (where the plaque is).  
     - The PINN’s color gradient (showing how displacement changes from left to right).

---

## 4. Why Use a PINN?

- Normally, neural networks just learn from **data**.  
- A **PINN** learns from data **and** from **physics equations**.  
- This means it can produce results that follow real-world laws (like elasticity or fluid flow), instead of just patterns in the data.

---

## 5. Step-by-Step Guide

1. **Install the Requirements**  
   - Use `pip install -r requirements.txt` to get all the libraries you need (like PyTorch, DeepXDE, and Streamlit).

2. **Put Your COCA Dataset in Place**  
   - Download the “COCA” dataset from Stanford AIMI.  
   - Place it in a folder called `dataset/cocacoronarycalciumandchestcts-2/Gated_release_final`, or update the path in `main.py`.

3. **Run the Main Script**  
   - Type `streamlit run main.py` in your terminal.  
   - If no models are found, the code trains a U‑Net on the CT images to find plaque. Then it trains the PINN to simulate how the bounding box would deform.

4. **Look at the Results**  
   - A web page will pop up (usually `http://localhost:8501`).  
   - You’ll see the **segmentation** of the plaque in blue on top of the CT scan.  
   - Below that, you’ll see a black-to-white gradient that shows the **displacement** from the PINN.

---

## 6. Technical Terms in Simple Words

- **DICOM:** A file format doctors use to store medical scans like CT or MRI.  
- **U‑Net:** A “U” shaped network that finds objects in images by encoding (downsampling) and decoding (upsampling) the image.  
- **Segmentation Mask:** A black-and-white image where white = “object of interest” and black = “not object.”  
- **PINN (Physics-Informed Neural Network):** A network that respects a physics equation (like elasticity) while it learns.  
- **PDE (Partial Differential Equation):** A math formula that says how things like stress, strain, or heat spread in space.  
- **Bounding Box:** The smallest rectangle that completely covers an object (in this case, the plaque).
