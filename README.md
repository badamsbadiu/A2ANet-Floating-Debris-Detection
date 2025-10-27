 A2ANet: Real-Time Detection of Floating Marine Debris Using Atrous Convolution and Channel Attention

 Overview
A2ANet is a lightweight, real-time deep learning architecture designed for floating marine debris detection using Atrous Convolution and Enhanced Channel Attention.  
The model achieves high detection accuracy while maintaining computational efficiency, enabling deployment on Unmanned Surface Vehicles (USVs) and low-power edge devices for ecological monitoring.

This repository contains the full implementation of A2ANet, including:
- Source code and model configuration files
- Training, validation, and inference scripts
- Dataset preprocessing and augmentation tools
- Example results and model weights



 Key Features
- Atrous Convolution for multi-scale context extraction  
- Enhanced Channel Attention (ECA) to suppress background noise and highlight debris objects  
- Optimized for real-time edge deployment (e.g., NVIDIA Jetson, GTX 1060)  
- Compatible with PyTorch ≥ 2.0 and the Ultralytics YOLOv5 framework


Repository Structure
A2ANet-Floating-Debris-Detection/
│
├── classify/ # Classification module (optional)
├── data/ # Dataset configs (.yaml), scripts, and augmentations
├── models/ # A2ANet model definition and architecture files
├── segment/ # Segmentation utilities (if used)
├── utils/ # Core functions (augmentation, dataloaders, logging)
│
├── train.py # Training script
├── val.py # Validation script
├── detect.py # Inference and visualization
├── requirements.txt # Package dependencies
└── README.md # Project documentation

 Installation
```bash
# Clone this repository
git clone https://github.com/badamsbadiu/A2ANet-Floating-Debris-Detection.git
cd A2ANet-Floating-Debris-Detection

# Create a Python environment
python -m venv a2anet_env
source a2anet_env/bin/activate   # (Linux/Mac)
a2anet_env\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt

Training
python train.py --data data/D_six.yaml --weights yolov5s.pt --batch-size 16 --epochs 500 --img 640

To reproduce the results from the paper, use:
Datasets: D_six, FloW-Img, and D_six + FloW-Img combined dataset
Resolution: 640 × 640
Batch Size: 16
Dilation Settings: Atrous [6] and [1, 2, 3]
Attention Module: Enhanced Channel Attention (ECA)

Evaluation
python val.py --weights runs/train/A2ANet/weights/best.pt --data data/FloW-Img.yaml --task test
Metrics reported:
mAP@0.5, mAP@0.5:0.95

Precision

Recall

Inference Speed (ms/frame)
Model Architecture
A2ANet introduces:

Multi-scale atrous convolution blocks at P1/2, P3/8, and P5/32 layers (dilation = 6)

Parallel atrous convolutions with dilation rates [1, 2, 3]

ECA attention block after spatial enrichment to prevent gridding and preserve fine features

These components jointly enhance small-object detection under complex aquatic lighting and background conditions.

Dataset
D_six Dataset

A custom, publicly available dataset for floating debris detection:

DOI: https://doi.org/10.5281/zenodo.15195086

Includes six debris categories (plastic bottles, styrofoam, plastic bag, plastic drink container, plastic take out, and can) collected from inland water surfaces under varying environmental conditions.

FloW-Img Dataset

Used for benchmarking, available from [Cheng et al., ICCV 2021].

Results Summary
| Dataset  | Model   | mAP@0.5   | mAP@0.5:0.95 | Recall    | Precision | Inference (ms) |
| -------- | ------- | --------- | ------------ | --------- | --------- | -------------- |
| D_six    | YOLOv5s | 0.787     | 0.498        | 0.714     | 0.879     | 36.0           |
| D_six    | A2ANet  | **0.841** | **0.533**    | **0.775** | 0.854     | 39.0           |
| FloW-Img | YOLOv5s | 0.883     | 0.441        | 0.788     | 0.896     | 7.9           |
| FloW-Img | A2ANet  | **0.892** | **0.449**    | **0.836** | 0.878     | 19.8           |

Citation

If you use this code or dataset, please cite:
Badams, B., & Co-Authors. (2025). 
A2ANet: Real-Time Detection of Floating Marine Debris Using Atrous Convolution and Channel Attention. 
*Ecological Informatics*. 
https://github.com/badamsbadiu/A2ANet-Floating-Debris-Detection


Data Availability

The dataset and code are publicly accessible:

Code: https://github.com/badamsbadiu/A2ANet-Floating-Debris-Detection

Dataset: https://doi.org/10.5281/zenodo.15195086

Weights: Will be released upon paper acceptance.

For full reproducibility, the repository includes:

Model training configuration files

Preprocessing scripts

Environment dependencies

License

This project is licensed under the MIT License — free to use and modify with attribution.

Acknowledgements

This work was developed as part of ongoing research at Universiti Teknologi Malaysia (UTM).
We thank contributors to the open-source YOLO framework and the Ecological Informatics community for advancing AI-driven environmental monitoring.
