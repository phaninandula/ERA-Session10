# 🌐 ERA1 Session 10 Assignment 🌐

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution](#Solution)
3. [Concepts discussed in this Session](#Concepts-discussed-in-this-Session)
4. [Training Status](#Training-Status)
5. [Results](#results)
6. [Classwise Accuracy](#classwise-accuracy)
7. [References](#References)

## 🎯 Problem Statement

Write a custom link to an external site. ResNet architecture for CIFAR10 that has the following architecture:

PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]  
Layer1 -  
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]  
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
Add(X, R1)  
Layer 2 -  
Conv 3x3 [256k]  
MaxPooling2D  
BN  
ReLU  
Layer 3 -  
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]  
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]  
Add(X, R2)  
MaxPooling with Kernel Size 4  
FC Layer  
SoftMax  
Uses One Cycle Policy such that:  
Total Epochs = 24  
Max at Epoch = 5  
LRMIN = FIND  
LRMAX = FIND  
NO Annihilation  
Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)  
Batch size = 512  
Use ADAM, and CrossEntropyLoss  
Target Accuracy: 90% 

NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training.  
Once done, proceed to answer the Assignment-Solution page.  

## 📚 Solution

The goal of this assignment is to design a Convolutional Neural Network (CNN) using PyTorch with certain requirements as below:
### 1. Data Augmentation: 
RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)  
### 2. Model Architecture:
The model should have an architecture as defined above in the problem statement. The objective is to use Residual blocks, OneCyclePolicy, Adam optimizer, cross-entropy loss, batch size of 512, and achieve 90% accuracy. The code for this assignment is provided in a Jupyter Notebook, which can be found [here](./ERA1_S10_CIFAR10_Resnet.ipynb).

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The 10 classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is divided into 50,000 training images and 10,000 validation images.
### 3. LRFinder
The below code is used to find the `max_lr`. From the results `max_lr is coming to be 5.34E-3`

<img width="656" alt="Screenshot 2023-07-21 at 10 21 30 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/63f2bd45-7584-488d-a5dd-d46e7d87e18c">

This 5.4e-3 value is set as `max_lr` in the OneCycleLR scheduler

<img width="1077" alt="Screenshot 2023-07-21 at 10 23 12 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/3fceb75a-48fb-4693-aaff-f8a9006ce2b5">

## Concepts discussed in this Session

#### 'RESNETs - Deep Residual Learning For Image Recognition by He et.al'
| What Problem Resnet is addressing?|           Background              |        Building blocks of ResNet  |
|-----------------------------------|-----------------------------------|-----------------------------------|
|<img width="569" alt="Screenshot 2023-07-21 at 5 30 46 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/80a220c9-931a-4e75-84dc-c19fc36e1572">|<img width="571" alt="Screenshot 2023-07-21 at 5 34 47 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/5cf96a34-9123-4388-9696-a0dd9d99e580">|<img width="507" alt="Screenshot 2023-07-21 at 5 40 06 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/6f328796-345a-49c1-9f04-11c123f3a5fa">|

| VGG-19 Vs Plain Network-34 layers Vs ResNet-34|    Results & Observations              |        Different types of shotcuts  |
|-----------------------------------|-----------------------------------|-----------------------------------|
|<img width="703" alt="Screenshot 2023-07-21 at 5 44 39 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/e2fb59cd-ec17-4664-8d13-edd1b9f73564">|<img width="600" alt="Screenshot 2023-07-21 at 5 45 10 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/f1c66c4f-b528-450f-acd3-f6601afa2088">|<img width="565" alt="Screenshot 2023-07-21 at 5 45 40 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/cf92b515-519e-43a8-ac60-cc59fd7e4f23">|

| Non-Bottleneck Vs Bottleneck Residual connection|    GAP reduces num of channels keeping width & height same              |        1x1 conv  |
|-----------------------------------|-----------------------------------|-----------------------------------|
|  <img width="568" alt="Screenshot 2023-07-21 at 5 45 57 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/36714576-ee0b-4fc4-b9ea-84108f59f397">| ![GAP](https://github.com/phaninandula/ERA-Session10/assets/30425824/d3c451e5-296d-4217-91e1-3cde3430f711)| ![1cross1](https://github.com/phaninandula/ERA-Session10/assets/30425824/ee708ad6-a9d0-419c-a204-bc9e62145e74)|

### OneCycleLR 

| LR range Test |    1CycleLR Calculation              |        Pytorch implementation  |
|-----------------------------------|-----------------------------------|-----------------------------------|
|![intro - 1cyclepolicy](https://github.com/phaninandula/ERA-Session10/assets/30425824/cc5eb2e6-6ba5-4218-8124-12f8216cc940)| ![1cyclepolicy calc](https://github.com/phaninandula/ERA-Session10/assets/30425824/a50fce2f-de08-425b-b9f4-aa7da5dd73e8)| ![onecycle_pytorch](https://github.com/phaninandula/ERA-Session10/assets/30425824/897bafbf-9bfe-4382-8c34-c7b76b982959)|

## Training Status (Logs)
<img width="1077" alt="Screenshot 2023-07-21 at 10 23 12 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/3c10c3fb-62c3-46f9-9258-92f918b1d64c">

## 📈 Results

The model was trained for 24 epochs and achieved an accuracy of 91.51% on the test set. The total number of parameters in the model was under 6573k. The training logs, as well as the output of the torch summary, are included in the notebook.

Training accuracy: 94.486 %
Test accuracy: 91.51 %

## 📊 Classwise Accuracy

<img width="523" alt="Screenshot 2023-07-21 at 10 26 34 PM" src="https://github.com/phaninandula/ERA-Session10/assets/30425824/ec865834-66ff-4419-9fff-23e4f114dde3">

## References
1. Deep Residual Learning for Image Recognition - He et.al 2015
2. Cyclical Learning Rates for Training Neural Networks - L.N.Smith - 2015
3. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates - L.N.Smith et.al 2018
4. Acknowledge all the authors who helped me understand the concepts in Medium. Couldnt save the links.
