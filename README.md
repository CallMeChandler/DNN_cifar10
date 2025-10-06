# 🧠 CIFAR-10 Deep Neural Network — From Scratch (NumPy) + Keras Benchmark

This project implements a **Deep Neural Network (DNN / MLP)** trained on the **CIFAR-10** image dataset — both **from scratch using pure NumPy** and using **TensorFlow/Keras** for benchmarking.

---

## 🚀 Results Overview

| Model | Framework | Input | Test Accuracy |
|--------|------------|--------|---------------|
| MLP (from scratch) | NumPy | 32×32 RGB/Gray (flattened) | **43%** |
| MLP | Keras / TensorFlow | 32×32 RGB/Gray (flattened) | **52%** |

> CIFAR-10 is a *core CNN dataset*, and MLPs lack spatial inductive bias — so 40–60% accuracy is typical before overfitting.  
> Hitting 43% manually (vs 52% in Keras) validates the math implementation and training loop.

---

## 🧩 What I Built

- Full **forward & backward propagation** for a deep MLP:  
  `Affine → ReLU → Affine → ReLU → Affine → Softmax`
- **Stable softmax** (log-sum-exp trick) + **cross-entropy loss**
- **He initialization**, **SGD + Momentum**, **L2 regularization**, **Early stopping**
- Clean **CIFAR-10 data pipeline** for both grayscale & RGB modes
- Comparative implementation using **Keras/TensorFlow**

---

## 📊 Key Visuals

| Visualization | Description |
|---------------|-------------|
| ![accuracy](images/accuracy_curve.png) | Training & Validation Accuracy |
| ![loss](images/loss_curve.png) | Training & Validation Loss |
| ![confusion](images/confusion_matrix.png) | Confusion Matrix |
| ![preds](images/sample_predictions.png) | Random 16-image predictions (✅ / ❌) |

*(Replace the image filenames with your actual screenshots — e.g., inside `/images/` or `/assets/` folder.)*

---

## 🧠 Dataset

**CIFAR-10** — 60,000 color images (32×32×3) in 10 classes  
- 50,000 training, 10,000 test images  
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
- Experiments were done on both **grayscale** (1×32×32) and **RGB** (3×32×32) variants.

---

## 🧮 Project Structure

├── 01_data_cifar10.ipynb # Data download, preprocessing, caching
├── 02_mlp_from_scratch.ipynb # MLP math (NumPy) — forward + backward pass
├── 03_train_eval_cifar10.ipynb # Training loop, early stopping, confusion matrix
├── 04_DNN_Keras_cifar10.ipynb # Keras/TensorFlow benchmark
├── data/ # CIFAR-10 cache (.npz)
└── images/ # Screenshots/plots (optional)

yaml
Copy code

---

## 🔬 Highlights

- Implemented the full training loop manually (no frameworks for autodiff)
- Verified gradients and loss numerically
- Reached **43% test accuracy** purely via NumPy math  
- Achieved **52%** using equivalent architecture in Keras
- Fun stat: random batch of 16 images → **12/16 predictions correct (75%)**

---

## 💡 Learnings

- **MLPs struggle on image data** because they flatten spatial patterns  
- **CNNs outperform** here due to local receptive fields & weight sharing  
- Even simple things like **standardization** or **He initialization** drastically impact convergence  
- Debugging matrix dimensions and gradient flow teaches you *real deep learning* far beyond auto-diff frameworks

---

## 📁 Repository

🔗 **GitHub:** [CallMeChandler/DNN_cifar10](https://github.com/CallMeChandler/DNN_cifar10)

---

## 🏁 Built With

- Python 3.10+  
- NumPy  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras

---

## 🧍‍♂️ Author

**Aakarsh Agarwal (CallMeChandler)**  
Learning Deep Learning the hard way — one gradient at a time.

---
