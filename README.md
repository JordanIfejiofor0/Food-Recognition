# Food-Recognition
This one is worth presenting as a proper **computer vision project**, because you weren't just using a pretrained ResNet or something—you designed the CNN architecture yourself and built the training pipeline around it.

# 🍔 Food Recognition CNN

A custom image classification model built with **PyTorch** that recognizes different types of fast food from images.

The project includes a custom CNN architecture, an image augmentation pipeline, MixUp training, label smoothing, mixed-precision training, and a Streamlit frontend for real-time image classification.

---

## Overview

The model takes an image of food and predicts which of **10 food classes** it belongs to.

```text
Image
  ↓
Resize to 64×64
  ↓
Custom CNN
  ↓
Feature Extraction
  ↓
Global Average Pooling
  ↓
Linear Classifier
  ↓
Food Class
```

The final model can recognize:

* 🥔 Baked Potato
* 🍔 Burger
* 🍗 Crispy Chicken
* 🍩 Donut
* 🍟 Fries
* 🌭 Hot Dog
* 🍕 Pizza
* 🥪 Sandwich
* 🌮 Taco
* 🌯 Taquito

---

## Demo

The project includes a Streamlit frontend where an image can be uploaded and classified.

```text
┌─────────────────────────────────┐
│        🍔 Food Classifier        │
│                                 │
│     Upload an image             │
│                                 │
│        [ Choose File ]          │
│                                 │
│          ↓                      │
│                                 │
│       🖼️ Uploaded Image         │
│                                 │
│   Predicted Food: Pizza 🍕      │
└─────────────────────────────────┘
```

---

# Model Architecture

Rather than using a pretrained computer vision model, I built the CNN architecture myself in PyTorch.

The model is based around convolutional blocks combining:

* Pointwise expansion
* Depthwise convolution
* Squeeze-and-Excitation
* Pointwise projection
* Batch normalization
* SiLU activation
* Residual connections

The overall architecture is:

```text
Input
  │
  ▼
3 × 64 × 64
  │
  ▼
Block: 3 → 32
  │
  ▼
Block: 32 → 64
  │
  ▼
Block: 64 → 64
  │
  ▼
Block: 64 → 128
  │
  ▼
Block: 128 → 128
  │
  ▼
Block: 128 → 256
  │
  ▼
Block: 256 → 256
  │
  ▼
Block: 256 → 512
  │
  ▼
Global Average Pooling
  │
  ▼
Dropout
  │
  ▼
Linear Layer
  │
  ▼
10 Classes
```

---

# Convolutional Blocks

The main building block of the network uses an inverted-bottleneck style design.

```text
Input
  │
  ▼
1×1 Expansion
  │
  ▼
BatchNorm + SiLU
  │
  ▼
3×3 Depthwise Convolution
  │
  ▼
BatchNorm + SiLU
  │
  ▼
Squeeze-and-Excitation
  │
  ▼
1×1 Projection
  │
  ▼
BatchNorm
  │
  ├───────────────┐
  │               │
  │          Skip Connection
  │               │
  └─────── + ─────┘
          │
          ▼
        SiLU
```

The expansion ratio is:

```text
expand_ratio = 6
```

This allows the block to temporarily increase the feature dimension before applying the depthwise convolution.

---

# Depthwise Convolution

The convolutional blocks use depthwise convolutions:

```python
nn.Conv2d(
    hidden_dim,
    hidden_dim,
    kernel_size=3,
    stride=stride,
    padding=1,
    groups=hidden_dim
)
```

Because each channel is convolved independently, depthwise convolution is considerably cheaper than applying a standard convolution across all input channels.

This allows the model to perform spatial feature extraction without making the network unnecessarily expensive.

---

# Squeeze-and-Excitation

Each block also contains a **Squeeze-and-Excitation (SE)** module.

The SE block uses global average pooling to summarize each feature channel and learns how important each channel is.

```text
Feature Maps
     │
     ▼
Global Average Pooling
     │
     ▼
Channel Reduction
     │
     ▼
SiLU
     │
     ▼
Channel Expansion
     │
     ▼
Sigmoid
     │
     ▼
Channel Weights
     │
     ×
     │
Original Features
```

The reduction ratio used is:

```text
r = 8
```

This allows the network to dynamically emphasize useful visual features.

---

# Residual Connections

The blocks use skip connections:

```python
return self.act(
    self.conv_block(x) + self.skip(x)
)
```

When the input and output dimensions are different, the skip connection uses a `1×1` convolution to match the dimensions.

Residual connections help information flow through deeper networks and make optimization easier.

---

# Training

The model was trained using:

```text
Optimizer: AdamW
Learning Rate: 5 × 10⁻⁴
Weight Decay: 1 × 10⁻⁴
Epochs: 40
Batch Size: 64
```

The learning rate was controlled using cosine annealing:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs
)
```

---

# Data Augmentation

The training pipeline uses several augmentations to improve generalization.

```python
transforms.RandomResizedCrop(
    64,
    scale=(0.6, 1.0)
)

transforms.RandomHorizontalFlip(p=0.5)

transforms.RandomRotation(15)

transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05
)

transforms.RandomErasing(
    p=0.25,
    scale=(0.02, 0.15)
)
```

This means the model sees slightly different versions of training images instead of memorizing the exact training examples.

The validation pipeline does not use random augmentation:

```text
Resize → Tensor → Normalize
```

---

# MixUp

One of the techniques used during training was **MixUp**.

Two training images are combined using a randomly sampled mixing coefficient:

```text
Mixed Image = λ × Image A + (1 − λ) × Image B
```

The corresponding targets are also mixed when calculating the loss:

```python
loss = (
    lam * loss(outputs, targets_a)
    + (1 - lam) * loss(outputs, targets_b)
)
```

The MixUp parameter used was:

```text
alpha = 0.4
```

This encourages the model to learn smoother decision boundaries and can help reduce overfitting.

---

# Label Smoothing

The classification loss uses label smoothing:

```python
nn.CrossEntropyLoss(
    label_smoothing=0.1
)
```

Instead of treating the correct class as having a probability of exactly `1.0`, label smoothing slightly distributes probability across the other classes.

This can help prevent the model from becoming overly confident in its predictions.

---

# Mixed Precision Training

Training also uses PyTorch's automatic mixed precision:

```python
with autocast(device_type="cuda"):
    outputs = net(inputs)
    loss = ...
```

and gradient scaling:

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This allows compatible operations to use lower-precision arithmetic while maintaining training stability through gradient scaling.

The goal was to make training faster and reduce GPU memory usage.

---

# Input Processing

Images are resized to:

```text
64 × 64
```

and normalized using:

```text
Mean = (0.5, 0.5, 0.5)
Std  = (0.5, 0.5, 0.5)
```

During inference, the image follows:

```text
Uploaded Image
      ↓
RGB Conversion
      ↓
64 × 64 Resize
      ↓
Tensor
      ↓
Normalization
      ↓
CNN
```

---

# Streamlit Frontend

The project includes a simple Streamlit interface.

The user uploads an image:

```python
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)
```

The image is then passed through the trained model.

The class with the highest output logit is selected:

```python
_, pred = torch.max(output, 1)
```

The corresponding class name is then displayed to the user.

---

# Model Saving

The trained model is saved together with the class mapping:

```python
model_data = {
    "model_state_dict": net.state_dict(),
    "class_to_idx": train_data.class_to_idx
}
```

This allows the model weights and dataset class mapping to be stored together.

The resulting checkpoint is:

```text
foodv2_model.pth
```

---

# Dataset

The model was trained using the **Fast Food Classification V2** dataset.

The dataset is structured using `torchvision.datasets.ImageFolder`, allowing the class directories to automatically define the classification labels.

The project uses:

```text
Train/
Valid/
```

for training and validation respectively.

---

# Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* PIL
* Streamlit
* CUDA

---

# What I Learned

This was one of my earlier computer vision projects and helped me move from basic CNN experimentation into designing more sophisticated architectures and training pipelines.

Some of the main concepts I worked with were:

* CNN architecture design
* Residual connections
* Depthwise convolutions
* Inverted bottleneck blocks
* Squeeze-and-Excitation
* Global average pooling
* Data augmentation
* MixUp
* Label smoothing
* AdamW
* Cosine learning-rate scheduling
* Mixed-precision training
* Model checkpointing
* Image classification
* Deploying a PyTorch model through Streamlit

The project also taught me that improving a vision model isn't just about making it deeper or wider. **Architecture, regularization, augmentation, optimization, and the data pipeline all matter.**

---

# Project Structure

```text
Food-Recognition/
│
├── v2/
│   └── foodv2_model.pth
│
├── modelv2.py
├── app.py
└── README.md
```

---

# Running the Project

Install the required packages:

```bash
pip install torch torchvision streamlit pillow numpy
```

Then start the Streamlit application:

```bash
streamlit run app.py
```

Upload an image and the model will return one of the ten supported food classes.

---

# Future Improvements

Some possible improvements would be:

* Add prediction confidence scores
* Display the top-3 predictions
* Train at a higher image resolution
* Add more food categories
* Experiment with larger CNN architectures
* Compare the custom model against pretrained architectures
* Add GPU inference
* Deploy the application publicly
* Add Grad-CAM visualizations to understand what the model focuses on

---

## Author

Built in **PyTorch** as a computer vision project focused on learning how to design and train a CNN from scratch and deploy it through a simple web interface.
