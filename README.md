# Improving-Neural-Network-Performance-Through-Hyperparameter-Tuning
Developed a Fashion-MNIST classifier using TensorFlow/Keras and performed hyperparameter tuning on neurons, batch size, network depth, regularization, and normalization, achieving up to 87.84% test accuracy.

##  Project Overview

This project investigates how different neural network design choices and hyperparameters affect the performance of a neural network on the **Fashion-MNIST image classification dataset**.

The main objective was not simply to build a neural network, but to systematically experiment with important hyperparameters and compare their effects on model performance. The study follows a **before-and-after ablation approach**, where different configurations are trained and evaluated to identify which choices improve or reduce classification performance.

The experiments focus on:

* Number of neurons
* Batch size
* Number of hidden layers
* Regularization
* Data normalization

Each experiment was evaluated using **test accuracy, test loss, training time, and validation learning curves**.

The best configuration achieved a **test accuracy of 87.84%**.


##  Objectives

The main objectives of this project were to:

1. Build a baseline neural network for Fashion-MNIST classification.
2. Understand the effect of different numbers of neurons.
3. Investigate how batch size influences model performance.
4. Study the effect of increasing network depth.
5. Analyze the impact of L1/L2 regularization.
6. Compare model performance before and after data normalization.
7. Use visualizations and performance metrics to compare different configurations.
8. Identify configurations that provide better predictive performance.

---

##  Dataset

The project uses the **Fashion-MNIST dataset**, which contains grayscale images belonging to 10 different clothing categories.

Each image has a resolution of **28 × 28 pixels**.

### Classes

The dataset contains the following 10 classes:

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

The original training data was divided into:

* **54,000 training samples**
* **6,000 validation samples**
* **10,000 test samples**

A fixed random seed (`42`) was used to make the experiments reproducible.

---

##  Data Preprocessing

Before training the neural network, the image data was preprocessed in two main steps.

### 1. Flattening

The original images have a shape of:

```text
28 × 28
```

Since the project uses a fully connected neural network, each image was flattened into a one-dimensional vector:

```text
28 × 28 = 784 features
```

Therefore, every image is represented by **784 input features**.

### 2. Normalization

Pixel values originally range from:

```text
0 to 255
```

They were normalized by dividing by `255.0`, resulting in values between:

```text
0 and 1
```

This preprocessing was applied separately to the training, validation, and test sets.

---

##  Neural Network Architecture

A flexible neural network architecture was implemented using **TensorFlow/Keras**.

The model consists of:

```text
Input Layer
     ↓
Dense Hidden Layer(s) + ReLU
     ↓
Optional Dropout
     ↓
Output Layer + Softmax
```

The input layer receives **784 features**, while the output layer contains **10 neurons**, one for each Fashion-MNIST class.

The hidden layers use the **ReLU activation function**, while the final layer uses **Softmax** to produce class probabilities.

The model was compiled using:

* **Optimizer:** Adam
* **Loss:** Sparse Categorical Crossentropy
* **Metric:** Accuracy

The architecture was implemented so that the number of hidden layers, neurons, dropout rate, and L1/L2 regularization could be changed easily for different experiments.

---

#  Experiments

## Experiment 1: Number of Neurons

The first experiment investigated how the number of neurons in the hidden layer affects classification performance.

The following configurations were tested:

```text
16, 32, 64, 128, 256, 512 neurons
```

### Results

| Neurons | Test Accuracy | Test Loss | Training Time |
| ------: | ------------: | --------: | ------------: |
|      16 |        85.18% |    0.4255 |        54.7 s |
|      32 |        86.01% |    0.3979 |        60.5 s |
|      64 |    **86.51%** |    0.4077 |        79.0 s |
|     128 |        86.38% |    0.4121 |        94.0 s |
|     256 |        85.38% |    0.4658 |       126.3 s |
|     512 |        85.63% |    0.4870 |       172.3 s |

The best result in this experiment was obtained with **64 neurons**, achieving **86.51% test accuracy**.

An important observation is that increasing the number of neurons does not always guarantee higher accuracy. After 64 neurons, performance decreased in this particular experiment, while training time continued to increase.

---

## Experiment 2: Batch Size

The project also investigated the effect of different batch sizes on neural network training.

Batch size controls how many training samples are processed before the model updates its weights.

Different batch-size configurations were evaluated while comparing their resulting performance.

The experiment showed that selecting an appropriate batch size can improve model performance without simply increasing model complexity.

The best configuration from the overall experiments achieved:

**87.84% test accuracy.**

---

## Experiment 3: Number of Hidden Layers

The third experiment investigated whether increasing the depth of the neural network improves classification performance.

The following architectures were tested:

```text
1 layer
2 layers
3 layers
4 layers
5 layers
```

### Results

| Hidden Layers | Test Accuracy |
| ------------: | ------------: |
|             1 |        86.39% |
|             2 |    **87.75%** |
|             3 |        86.80% |
|             4 |        85.57% |
|             5 |        86.56% |

The **2-hidden-layer configuration** achieved the highest accuracy in this experiment at **87.75%**.

The results demonstrate that adding more layers does not automatically improve performance. In this experiment, increasing the depth beyond two layers resulted in lower test accuracy.

---

## Experiment 4: Regularization

Regularization was investigated to understand its effect on model generalization.

The project included support for:

* **L1 regularization**
* **L2 regularization**
* **L1 + L2 regularization**
* Dropout

The purpose of regularization is to control model complexity and reduce overfitting.

The before-and-after comparison showed that regularization did not improve the final test accuracy in this particular set of experiments. However, it provided an opportunity to analyze the trade-off between model complexity, generalization, and predictive performance.

This demonstrates an important point of ablation studies: **a technique that is generally useful may not necessarily improve the test accuracy for every dataset or configuration.**

---

## Experiment 5: Data Scaling / Normalization

The project also compared model performance with and without appropriate input scaling.

The original image pixel values range from 0 to 255. The normalized version scales these values to the range:

```text
0 to 1
```

The comparison showed a clear improvement after normalization.

### Result

```text
Before Normalization: 83.87%
After Normalization:  87.08%

Improvement: +3.21 percentage points
```

This was one of the most noticeable improvements observed in the study.

---

#  Baseline Model

A baseline neural network was first trained before performing the different experiments.

### Baseline Configuration

```text
Hidden Layers: 1
Neurons: 128
Dropout: 0
Batch Size: 32
Epochs: 15
```

### Baseline Performance

```text
Test Accuracy: 87.63%
Test Loss:     0.3855
Training Time: 93.8 seconds
```

The baseline provided a reference point for comparing the subsequent experiments.

---

#  Overall Results

The major before-and-after comparisons from the study are summarized below:

| Experiment        | Before / Worst | After / Best |       Change |
| ----------------- | -------------: | -----------: | -----------: |
| Number of Neurons |         85.18% |       86.51% | **+1.33 pp** |
| Batch Size        |         85.68% |       87.84% | **+2.16 pp** |
| Hidden Layers     |         85.57% |       87.75% | **+2.18 pp** |
| Regularization    |         88.48% |       87.48% | **-1.00 pp** |
| Normalization     |         83.87% |       87.08% | **+3.21 pp** |

###  Best Overall Accuracy

The highest test accuracy obtained during the experiments was:

# **87.84%**

This result came from the optimized batch-size configuration.

---

#  Evaluation and Visualization

Several functions were developed to make the experimental comparison easier.

### Validation Curves

Validation accuracy and validation loss were plotted across epochs to observe how different configurations learned over time.

### Bar Charts

Bar charts were used to compare test accuracy between different configurations.

### Results Tables

Results were organized into tables containing:

* Configuration
* Test Accuracy
* Test Loss
* Training Time

### Before-and-After Comparison

The project also automatically identified the worst and best configurations based on test accuracy and compared them using:

* Test accuracy
* Test loss
* Validation accuracy curves
* Validation loss curves

These visual comparisons make it easier to understand not only **which configuration performed better**, but also **how the learning behavior changed**.

---

#  Key Findings

Several important observations were obtained from the ablation study:

### 1. More neurons do not always mean better performance

Increasing the number of neurons increased computational cost, but the highest number of neurons did not produce the highest accuracy.

### 2. Network depth requires careful selection

The two-layer network performed better than the deeper configurations tested. Adding additional layers did not consistently improve accuracy.

### 3. Batch size can significantly influence performance

The experiments demonstrated that batch size is an important training hyperparameter and that an appropriate value can improve model performance.

### 4. Normalization was highly beneficial

Scaling the input pixel values from 0–255 to 0–1 produced a **3.21 percentage-point improvement** in the comparison.

### 5. Regularization involves a trade-off

Regularization can help control overfitting, but in these experiments it resulted in a decrease in test accuracy compared with the corresponding non-regularized configuration.

### 6. Hyperparameter tuning is dataset-dependent

There is no single hyperparameter setting that guarantees the best performance. The results depend on the architecture, dataset, preprocessing, and training configuration.

---

#  Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

---

#  Project Structure

A recommended GitHub repository structure is:

```text
Fashion-MNIST-Neural-Network-Tuning/
│
├── Improving_Accuracy_of_Neural_Networks.ipynb
├── README.md
└── images/
    ├── dataset_samples.png
    ├── neuron_comparison.png
    ├── batch_size_comparison.png
    ├── depth_comparison.png
    └── before_after_comparison.png
```

The notebook contains the complete implementation, experiments, evaluation, and visualizations.

---

#  How to Run the Project

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## 2. Open the notebook

Open:

```text
Improving_Accuracy_of_Neural_Networks.ipynb
```

You can run it using:

* Google Colab
* Jupyter Notebook
* JupyterLab

## 3. Install the required libraries

If required, install the main dependencies:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

## 4. Run the notebook

Execute the notebook cells sequentially.

The Fashion-MNIST dataset is loaded directly through Keras, so the dataset is downloaded automatically when required.

---

#  Reproducibility

A fixed random seed was used:

```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

This helps make the experimental results more reproducible across runs.

---


#  Conclusion

This project presents a systematic study of neural network performance using Fashion-MNIST. Instead of relying on a single model configuration, multiple hyperparameters were varied and evaluated through controlled experiments.

The study achieved a maximum test accuracy of **87.84%**, while demonstrating how neurons, batch size, network depth, regularization, and normalization can produce different effects on model performance.

The results highlight the importance of **data preprocessing, appropriate hyperparameter selection, and empirical experimentation** when developing neural network models.


---

##  Project Highlights

```text
Dataset          → Fashion-MNIST
Model            → Fully Connected Neural Network
Framework        → TensorFlow / Keras
Classes          → 10
Input Features   → 784
Baseline Accuracy→ 87.63%
Best Accuracy    → 87.84%
Experiments      → 5 Hyperparameter Factors
Seed             → 42
