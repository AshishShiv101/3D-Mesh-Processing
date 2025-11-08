# 🧠 3D Mesh Processing  
**Normalization, Quantization & Error Analysis Pipeline**  
[🔗 View Repository →](https://github.com/<your-username>/3D-Mesh-Processing)

---

![3D Mesh Screenshot](https://github.com/user-attachments/assets/your-screenshot-id-here)

A complete Python-based project for **3D mesh preprocessing**, normalization, quantization, and reconstruction error analysis — built for AI and graphics research.  
This project converts raw `.obj` mesh data into AI-ready formats and visualizes the impact of each transformation step.

---

## ✨ Features

### 🧩 Mesh Loading & Inspection  
- Load `.obj` meshes using **Trimesh**  
- Analyze vertex statistics: *min, max, mean, std*  

### ⚖️ Normalization Techniques  
- **Min–Max Normalization**  
- **Unit Sphere Normalization**  

### 🔢 Quantization & Dequantization  
- Convert continuous coordinates into **1024 discrete bins**  
- Reconstruct (denormalize) and measure precision loss  

### 📊 Error Analysis  
- Compute **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**  
- Generate **axis-wise bar plots** to visualize reconstruction accuracy  

### 🖼️ Visualizations  
- Before/After mesh screenshots using **Open3D**  
- Automatic rendering & 2D scatter projections  

### 🧠 Modular Workflow  
A clean, reproducible 3-step pipeline:
1. **Inspect** → Load & analyze mesh  
2. **Normalize** → Apply scaling transformations  
3. **Reconstruct** → Quantize, denormalize & evaluate error  

---

## 🧰 Tech Stack

| Category           | Tools & Libraries |
|--------------------|------------------|
| **Language**       | Python 3.12+ |
| **3D Processing**  | Trimesh, Open3D |
| **Math / Data**    | NumPy, Pandas, SciPy |
| **Visualization**  | Matplotlib, Open3D |
| **Utility**        | tqdm, pyglet |

---

## 🧮 Example Outputs

### Person Mesh (Min–Max Normalization)
<img width="900" alt="person_minmax" src="https://github.com/user-attachments/assets/3f5217d4-0a62-44ae-ad24-18996cb8e145" />

### Reconstructed Mesh (After Quantization)
<img width="900" alt="person_minmax_recon" src="https://github.com/user-attachments/assets/90668c03-970c-4bb9-9dbb-9c7b91a939bb" />

### Branch Mesh Example
<img width="900" alt="branch_minmax" src="https://github.com/user-attachments/assets/c38740b8-709c-45fe-b2c1-93bc0fb5b732" />

---

## 🧱 Directory Structure

