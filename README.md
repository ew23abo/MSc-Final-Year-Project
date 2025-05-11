# MSc-Final-Year-Project
Artificial Intelligence-Driven Convolutional Neural Networks (CNNs) for Histopathology Image Classification: Detecting Metastatic Breast Cancer Using PatchCamelyon (PCam)
# AI-Powered Histopathology Image Classification  
**Detecting Metastatic Breast Cancer Using CNNs on the PatchCamelyon (PCam) Dataset**

## Overview of project
This repository contains code and resources for a deep learning project focused on the automated classification of metastatic breast cancer from histopathology image patches. The study compares three Convolutional Neural Network (CNN) architectures:

- **DenseNet-121** (pretrained, fine-tuned)
- **EfficientNet-B0** (lightweight, scalable)
- **Custom MM-SEN-Inspired Model** (designed for interpretability and efficiency)

The models were trained and evaluated on the **PatchCamelyon (PCam)** dataset, leveraging data augmentation, transfer learning, fine-tuning, and model interpretation tools such as Grad-CAM.

## Dataset
**PatchCamelyon (PCam)**  
- Source: [TensorFlow Datasets - patch_camelyon](https://www.tensorflow.org/datasets/catalog/patch_camelyon)  
- 327,680 RGB images (96×96)  
- Binary labels: 0 = Non-metastatic, 1 = Metastatic  
- Open-source and anonymised, suitable for academic research.

## Features
- Custom data loading, preprocessing, and augmentation pipelines.
- Transfer learning and fine-tuning for DenseNet and EfficientNet.
- Lightweight custom CNN with attention mechanisms (Squeeze-and-Excitation).
- Evaluation via accuracy, precision, recall, F1 score, and AUC-ROC.
- Visualisations: training curves, confusion matrices, ROC curves, Grad-CAM.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pcam-cancer-classification.git
    cd pcam-cancer-classification
    ```

2. Set up your environment:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook:
    ```bash
    Open `main_notebook.ipynb` in Google Colab or Jupyter Notebook.
    ```

4. To test trained models:
    ```python
    model = tf.keras.models.load_model('model_name.h5')
    ```

## Results

| Model              | Accuracy % | Precision % | Recall % | F1 Score %| AUC-ROC % |
|-------------------|-------------|-------------|----------|-----------|-----------|
| DenseNet-121       | 95.45      | 95.69       | 95.18    | 95.43     | 95.45   |
| EfficientNet-B0    | 91.40      | 89.32       | 94.03    | 91.62     | 97.45   |
| MM-SEN (Custom)    | 79.94      | 82.46       | 76.01    | 79.10     | 79.93   |

## Visual Examples
- Confusion matrices
- ROC and PR curves
- Grad-CAM attention heatmaps

## License
This project is licensed under the MIT License.

## Author
**ew23abo**  
University of Hertfordshire  
MSc Data Science with Advanced Research 

## Acknowledgements
- [Ge et al., 2024] MM-SEN for tumor detection
- [Tan & Le, 2019] EfficientNet paper
- [Zhong et al., 2020] DenseNet for cancer classification
- [Veeling et al., 2018] PatchCamelyon dataset
- Supervisor: Dr. Ralf Napiwotzki, for continuous guidance and support.
- Personal Motivation: This research was inspired by the journey of overcoming breast cancer, highlighting the need for better diagnostic tools.
- Family Support: Special thanks to my parents and family for their unwavering support throughout this project.

