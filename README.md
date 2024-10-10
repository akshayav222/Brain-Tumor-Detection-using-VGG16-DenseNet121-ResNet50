# Brain Tumor Detection using VGG16, DenseNet121, and ResNet50

This repository contains a deep learning-based solution for detecting brain tumors using three state-of-the-art Convolutional Neural Network (CNN) architectures: VGG16, DenseNet121, and ResNet50. The models are trained on a brain tumor dataset, where the task is to classify MRI images as either tumor-positive or tumor-negative.

## Key Features

- **Multiple Model Architectures**: Implements VGG16, DenseNet121, and ResNet50 for comparison of performance across architectures.
- **Transfer Learning**: Pre-trained weights are used to fine-tune the models for better accuracy with less data.
- **Data Augmentation**: Implements various data augmentation techniques to enhance model generalization and address data imbalance.
- **Imbalanced Dataset Handling**: The dataset used is imbalanced, and techniques like class weighting or oversampling can be used to address this issue.
- **Performance Metrics**: Includes evaluation metrics accuracy to measure model performance.
- **Visualization**: Uses Grad-CAM for visualizing model attention on MRI scans, providing interpretability to the predictions.

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Brain-Tumor-Detection.git
    cd Brain-Tumor-Detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```bash
    python train_model.py
    ```

4. Evaluate the model:
    ```bash
    python evaluate_model.py
    ```

5. Visualize model predictions:
    ```bash
    python visualize_results.py
    ```

## Dataset

The dataset used consists of MRI brain images, categorized into two classes: **Tumor** and **No Tumor**. Due to the imbalance in class distribution, appropriate techniques such as class weighting or oversampling have been applied.

## Results

The results from each model are compared based on accuracy evaluation metrics. The model's performance can be improved by adjusting hyperparameters or further tuning the architectures.

## Future Improvements

- Experiment with other CNN architectures.
- Implement model ensembling to improve accuracy.
- Explore further data augmentation techniques and balancing strategies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

