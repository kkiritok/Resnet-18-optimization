# Selective Weight Optimization for ResNet-18

## Project – Neural Networks / Machine Learning

This repository contains the implementation of a **Selective Weight Optimization approach** applied to a pretrained ResNet-18 model for image classification.

The project focuses on improving CIFAR-100 classification performance while preserving the pretrained knowledge of the model. It uses selective weight updates, weight truncation, and gradient masking to optimize only a subset of network parameters.

The performance is evaluated using **CIFAR-100 and Mini-ImageNet datasets**.

---

## Team Members

- Kandregula Vishal Karthik – N210037
- Ramagiri Praneetha – N210477
- Kondru Ramya – N210627
- Arigela Pranathi – N210175

**Guide:** KK Sing Sir  
**Institution:** RGUKT

---

## Problem Statement

To implement a selective weight optimization technique for deep neural networks and analyze its impact on classification performance using a pretrained ResNet-18 model.

The main objective is to improve CIFAR-100 accuracy while maintaining the pretrained knowledge and feature representations of the model.

---

## Project Motivation

- Updating all network parameters requires significant computational resources.
- Unnecessary weight updates can affect previously learned features.
- Selective weight optimization updates only a small portion of model parameters.
- Weight truncation helps identify important weights for optimization.
- The approach aims to improve classification accuracy while preserving pretrained knowledge.

---

## Project Approach

### 1. Pretrained ResNet-18

- Uses an ImageNet-pretrained ResNet-18 model.
- CIFAR-100 is used as the primary training dataset.
- Mini-ImageNet is used for additional evaluation.
- Input images are resized to 224 × 224.

### 2. Selective Weight Optimization

- Randomly selects 10% of weights from each backbone layer.
- Weight truncation is performed using mean absolute weight values.
- Gradient masking ensures that only selected weights are updated.
- The remaining weights are kept unchanged.

### 3. Weight Update Mechanism

- The old weights and updated candidate weights are combined.
- Uses 90% of the previous weights and 10% of the candidate weights.
- This helps control large changes in pretrained parameters.

### 4. Iterative Optimization

- The model is optimized through multiple iterations.
- Each iteration selects a fresh random subset of weights.
- CIFAR-100 accuracy is used to guide the optimization process.
- Early stopping is used with a patience of 3 iterations.

### 5. Evaluation

- CIFAR-100 performance is evaluated using classification accuracy and cross-entropy loss.
- Mini-ImageNet performance is evaluated using accuracy, KL divergence, and prediction agreement.

---

## Datasets

### CIFAR-100
- 100 classes.
- 60,000 images.
- 50,000 training images and 10,000 testing images.
- Used as the primary training and optimization dataset.

### Mini-ImageNet
- 1000 classes in the dataset used in this project.
- Used for evaluating the preservation of pretrained knowledge.
- KL Divergence is used to compare teacher and student predictions.

---

## Experimental Setup

| Parameter | Configuration |
|---|---|
| Model | ResNet-18 |
| Pretrained Weights | ImageNet |
| Framework | PyTorch |
| Accelerator | GPU (Kaggle) |
| Batch Size | 64 |
| Optimizer | SGD |
| CIFAR-100 Loss | Cross Entropy |
| Mini-ImageNet Loss | KL Divergence |
| Selected Weights | 10% per backbone layer |
| Weight Blending | 90% old + 10% candidate |
| Maximum Iterations | 20 |
| Epochs per Iteration | 3 |
| Early Stopping Patience | 3 |

---

## Evaluation Metrics

- **Training Loss:** Measures the classification error during training.
- **CIFAR-100 Accuracy:** Measures the main classification performance.
- **Mini-ImageNet Accuracy:** Evaluates pretrained feature preservation.
- **KL Divergence:** Measures the difference between teacher and student predictions.
- **Prediction Agreement:** Measures how consistently teacher and student models predict the same classes.
- **Convergence:** Tracks model performance across optimization iterations.

---

## Technologies & Libraries Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- CUDA

---

## Repository Structure

```text
├── BaseCode.py
├── requirements.txt
├── README.md
└── results/
    ├── best_model.pth
    ├── final_model.pth
    ├── training_history.csv
    ├── final_training_history.csv
    └── final_summary.csv
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python BaseCode.py
```

---

## Key Takeaways

- Selective weight optimization reduces unnecessary parameter updates.
- Weight truncation helps identify weights for optimization.
- Gradient masking restricts updates to selected parameters.
- Weight blending helps maintain stability during optimization.
- CIFAR-100 is used to improve classification performance.
- Mini-ImageNet evaluation helps analyze the preservation of pretrained knowledge.

---

## Future Work

- Experiment with different weight selection percentages.
- Extend selective optimization to Batch Normalization parameters.
- Explore optimization on deeper ResNet architectures.
- Evaluate on larger and more diverse datasets.
- Investigate advanced weight selection strategies.

---

## Disclaimer

This project implements a selective weight optimization technique using classical deep learning methods. It does not require quantum hardware or quantum computing algorithms.

The implementation is intended for academic and educational purposes.

---

## Acknowledgements

- https://github.com/kkClasses/FSCIL_github/
- Faculty guidance and support.
- PyTorch and the open-source community.
- Research literature on selective weight optimization and transfer learning.
