# Chapter 5 Notes - The Pet Breeds

**Book:** Deep Learning for Coders with FastAI and PyTorch
**Chapter:** 5 - The Pet Breeds
**Status:** Completed

---

# Big Picture

This chapter explains how CNNs (Convolutional Neural Networks) work and why they are so effective for image recognition.

The overall CNN pipeline:

```text
Image
↓
Convolution
↓
Feature Map
↓
ReLU
↓
Pooling
↓
More CNN Blocks
↓
Classifier Head
↓
Prediction
```

---

# Why CNNs Exist

## Problem with Fully Connected Networks

Images contain a huge number of pixels.

Example:

```text
224 × 224 × 3
=
150,528 values
```

If every pixel connects to every neuron:

```text
Too many parameters
Too much computation
Overfitting risk
```

---

## Key Insight

Nearby pixels matter more than distant pixels.

Example:

A cat ear is formed by nearby pixels, not pixels from opposite sides of the image.

CNNs exploit this local structure.

---

# Convolution

A convolution looks at a small patch of an image instead of the whole image.

Example:

```text
Image Patch

1 2 3
4 5 6
7 8 9
```

Kernel:

```text
1 0 1
0 1 0
1 0 1
```

Process:

1. Multiply corresponding values
2. Sum everything
3. Produce one output number

Example:

```text
1×1 + 2×0 + 3×1
+
4×0 + 5×1 + 6×0
+
7×1 + 8×0 + 9×1

=
25
```

---

## Important Idea

Convolution is essentially a:

```text
Dot Product
```

between:

```text
Image Patch
and
Kernel
```

---

# Kernel (Filter)

A kernel is a small pattern detector.

Example:

```text
3×3 Kernel
```

The kernel slides across the image searching for patterns.

Possible learned patterns:

```text
Edges
Textures
Curves
Corners
Eyes
Fur Patterns
```

---

# Weight Sharing

The same kernel is reused everywhere in the image.

Instead of learning:

```text
Millions of weights
```

we learn:

```text
One kernel
```

and use it everywhere.

Benefits:

* Fewer parameters
* Faster training
* Better generalization

---

# Feature Maps

A feature map is the collection of all convolution outputs.

```text
Image
↓
Kernel
↓
Many Outputs
↓
Feature Map
```

Feature maps tell us:

```text
Where a feature exists
How strongly it exists
```

Large value:

```text
Strong pattern detected
```

Small value:

```text
Weak pattern detected
```

---

# Multiple Kernels

One kernel can only learn one type of feature.

Real images contain many features.

Example:

```text
Cat
├─ Ears
├─ Eyes
├─ Nose
├─ Fur
└─ Tail
```

Therefore CNNs use multiple kernels.

Rule:

```text
1 Kernel
=
1 Feature Map
```

Example:

```text
16 Kernels
=
16 Feature Maps
```

---

# Stride

Stride controls how far the kernel moves.

Stride = 1

```text
Move 1 pixel at a time
```

Stride = 2

```text
Jump 2 pixels at a time
```

---

## Tradeoff

Small Stride:

```text
More Detail
More Computation
```

Large Stride:

```text
Less Detail
Less Computation
```

---

# Padding

Problem:

Convolution shrinks images.

Example:

```text
5×5 Image
3×3 Kernel

Output:
3×3
```

After many convolutions:

```text
Information is lost
```

Solution:

Padding.

Add extra pixels around the border.

Example:

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

Benefits:

```text
Preserve spatial dimensions
Preserve edge information
Allow deeper CNNs
```

---

# ReLU

ReLU stands for:

```text
Rectified Linear Unit
```

Formula:

```python
ReLU(x) = max(0, x)
```

Examples:

```python
ReLU(-5) = 0
ReLU(-2) = 0
ReLU(3) = 3
ReLU(7) = 7
```

---

## Why ReLU Exists

Without ReLU:

```text
Linear
↓
Linear
↓
Linear

=
One Big Linear Model
```

ReLU introduces:

```text
Non-Linearity
```

which allows neural networks to learn complex patterns.

---

# Max Pooling

Max pooling reduces feature map size.

Example:

Input:

```text
1 4
7 2
```

Output:

```text
7
```

because:

```text
7 is the maximum value
```

---

## Why Pooling?

1. Reduce computation
2. Reduce memory usage
3. Keep strongest features
4. Make model robust to small shifts

Pooling has:

```text
No learnable parameters
```

---

# CNN Feature Hierarchy

This is the most important idea in CNNs.

```text
Pixels
↓
Edges
↓
Shapes
↓
Parts
↓
Objects
```

Example:

```text
Pixels
↓
Lines
↓
Curves
↓
Eyes
↓
Cat Face
↓
Cat
```

The concept of:

```text
Cat
```

appears in deep layers, not early layers.

---

# Transfer Learning

Instead of training from scratch:

```text
Pretrained Model
↓
Fine-Tune
↓
New Task
```

Example:

```python
vision_learner(dls, resnet34)
```

loads a pretrained model.

---

## Why Transfer Learning?

Benefits:

```text
Less Data Required
Less Computation
Faster Training
Better Performance
```

---

# Fine-Tuning

FastAI's:

```python
learn.fine_tune()
```

roughly performs:

```text
Freeze
↓
Train Head
↓
Unfreeze
↓
Train Whole Model
```

---

# Why Freeze First?

To avoid:

```text
Catastrophic Forgetting
```

The model already knows:

```text
Edges
Shapes
Textures
Objects
```

We don't want to destroy that knowledge immediately.

---

# ResNet

ResNet stands for:

```text
Residual Network
```

Example:

```python
resnet34
```

A CNN architecture with residual connections.

---

# The Problem ResNet Solves

Very deep networks suffer from:

```text
Vanishing Gradients
```

Training becomes difficult.

---

# Skip Connections

Normal Network:

```text
Input
↓
Layers
↓
Output
```

ResNet:

```text
Input
├──────────►
│
Layers
│
▼
+
│
▼
Output
```

The input can bypass layers.

---

# Residual Learning

Instead of learning:

```text
H(x)
```

ResNet learns:

```text
F(x)
```

and computes:

```text
Output = Input + F(x)
```

If:

```text
F(x) = 0
```

Then:

```text
Output = Input
```

This makes optimization easier.

---

# Key Exam Points

1. CNNs exploit local structure in images.
2. Convolution is essentially a dot product.
3. Kernels are pattern detectors.
4. One kernel produces one feature map.
5. ReLU introduces non-linearity.
6. Pooling reduces feature map size.
7. CNNs learn hierarchical features.
8. Transfer learning reuses pretrained knowledge.
9. Fine-tuning = Freeze → Train Head → Unfreeze.
10. ResNet uses skip connections.
11. Skip connections help gradients flow.
12. ResNet solves deep network training problems.

---

# Personal Understanding

My current mental model:

```text
Image
↓
Convolution
↓
Feature Map
↓
ReLU
↓
Pooling
↓
More CNN Blocks

Pixels
↓
Edges
↓
Shapes
↓
Parts
↓
Objects

Classifier Head
↓
Prediction
```

This is the core idea of Chapter 5.
