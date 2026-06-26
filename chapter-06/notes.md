# Chapter 6 Notes – Multi-Label Classification & Regression (FastAI)

> **Book:** Deep Learning for Coders with FastAI and PyTorch
> **Goal:** Understand the concepts instead of memorizing code.

---

# Chapter Overview

Chapter 6 introduces two new computer vision tasks:

1. **Multi-label Classification**
2. **Key Point Regression**

Unlike Chapter 5, where every image belonged to **one class**, this chapter teaches how to predict **multiple labels** and **continuous coordinates**.

---

# 1. Classification vs Multi-label Classification vs Regression

## Classification

One image → One label

Example

```
Image
   ↓
Dog
```

Output:

```
Dog
```

Examples:

* Cat vs Dog
* Pet Breeds
* MNIST digits

---

## Multi-label Classification

One image → Multiple labels

Example

```
Image

Dog
Person
Chair
```

Output:

```
Dog
Person
Chair
```

An image can belong to multiple classes simultaneously.

Examples:

* Object Detection (without boxes)
* Scene Recognition
* Medical Images

---

## Regression

One image → Numbers

Example

```
Face Image
```

Output

```
(x, y)
```

Instead of predicting categories, the model predicts continuous values.

Examples:

* Face keypoints
* Steering angle
* House price
* Temperature

---

# 2. Multi-label Classification

## Problem

Previous chapter:

```
Dog Image

↓

Beagle
```

Chapter 6:

```
Street Image

↓

Person
Car
Traffic Light
```

One image can contain many objects.

---

# Pascal VOC Dataset

Jeremy uses the Pascal VOC dataset.

Each image may contain multiple objects.

Example

```
Image

Dog
Chair
Person
```

Labels:

```
dog chair person
```

---

# Dataset Structure

Unlike Pet Breeds,

Labels are NOT stored in folders.

Instead:

```
train.csv

Image Name
Labels
Validation Flag
```

Example

| Image | Labels    | is_valid |
| ----- | --------- | -------- |
| img1  | dog chair | False    |
| img2  | person    | True     |

---

# Constructing a DataBlock

```python
dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    splitter=splitter,
    get_x=get_x,
    get_y=get_y,
    item_tfms=RandomResizedCrop(128)
)
```

Think of DataBlock as a recipe.

It tells fastai:

* What is the input?
* What is the output?
* Where are images?
* Where are labels?
* How to split data?
* What preprocessing to apply?

---

## ImageBlock

Represents the input.

```
Image

↓

Tensor
```

---

## MultiCategoryBlock

Represents multiple labels.

Instead of

```
Dog
```

it supports

```
Dog
Chair
Person
```

---

## get_x

Purpose:

Find image.

Example

```
CSV Row

↓

image path
```

Returns

```
.../images/000123.jpg
```

---

## get_y

Purpose:

Find labels.

Example

```
dog chair person
```

Returns

```
['dog','chair','person']
```

---

## splitter

Separates

```
Training Set

Validation Set
```

Jeremy uses the official dataset split instead of RandomSplitter.

---

## DataLoaders

```python
dls = dblock.dataloaders(df)
```

This converts the recipe into actual batches.

```
CSV

↓

Images

↓

Labels

↓

DataLoader

↓

Mini Batches
```

---

# Multi-hot Encoding

Suppose classes are

```
Dog
Cat
Chair
Person
```

Image contains

```
Dog
Person
```

Encoding

```
[1,0,0,1]
```

Unlike one-hot encoding, multiple positions can be 1.

---

# Neural Network Output

```python
activs = learn.model(x)
```

Output shape

```
torch.Size([64,20])
```

Meaning

```
64 images

20 class scores
```

Each image receives one score for every class.

---

# Logits

The model outputs raw numbers.

Example

```
[-3.1,
 2.8,
 0.5,
-1.7]
```

These are called

```
Logits
```

They are NOT probabilities.

---

# Sigmoid

Sigmoid converts logits into probabilities.

```
Logit

↓

Sigmoid

↓

Probability
```

Example

```
5.0

↓

0.993
```

```
-5

↓

0.007
```

Range

```
0 → 1
```

---

# Binary Cross Entropy (BCE)

Purpose:

Measure how wrong predictions are.

Simple definition:

> Binary Cross Entropy compares predicted probabilities with the true labels and penalizes incorrect predictions.

Lower BCE

↓

Better model

Higher BCE

↓

Worse model

---

## BCEWithLogitsLoss

Instead of

```python
sigmoid()

↓

Binary Cross Entropy
```

PyTorch combines both into

```python
nn.BCEWithLogitsLoss()
```

Advantages

* Faster
* Numerically stable
* Used in practice

---

# Threshold

Model predicts

```
0.83
0.12
0.67
0.05
```

Need to decide which classes exist.

Example

Threshold = 0.2

```
0.83 ✓

0.12 ✗

0.67 ✓

0.05 ✗
```

Predicted labels

```
[1,0,1,0]
```

Changing threshold changes accuracy.

---

# accuracy_multi

Used instead of normal accuracy.

```python
metrics=partial(accuracy_multi, thresh=0.2)
```

Because every image may have multiple correct answers.

---

# Fine Tuning

```python
learn.fine_tune(3)
```

Stage 1

Freeze pretrained backbone.

Train only classifier.

Stage 2

Unfreeze entire model.

Train everything.

Benefits

* Faster
* Better accuracy
* Less data required

---

# Key Point Regression

Instead of predicting categories,

the model predicts coordinates.

Example

```
Image

↓

(0.35,-0.42)
```

This is called regression.

---

# Why Regression?

Categories are finite.

```
Dog

Cat

Horse
```

Coordinates are continuous.

```
0.271

0.314

0.752

...
```

Infinite possibilities.

---

# img2pose()

Purpose

Convert

```
image.jpg
```

into

```
image_pose.txt
```

so fastai can locate the annotation file.

---

# get_ctr()

Purpose

Read the pose file.

Convert 3D coordinates into 2D image coordinates.

Return

```
tensor([x,y])
```

These become the labels.

---

# y_range

```python
learn = vision_learner(
    dls,
    resnet18,
    y_range=(-1,1)
)
```

Purpose

Restrict model outputs between

```
-1

and

1
```

Useful because coordinates were normalized into that range.

---

# sigmoid_range

Normal sigmoid

```
0 → 1
```

sigmoid_range

```
-1 → 1
```

Fastai uses this internally for regression.

---

# Complete Multi-label Pipeline

```
CSV

↓

DataBlock

↓

DataLoaders

↓

ResNet

↓

Logits

↓

Sigmoid

↓

Threshold

↓

Predicted Labels
```

---

# Complete Regression Pipeline

```
Image

↓

DataBlock

↓

DataLoader

↓

ResNet

↓

(x,y)

↓

Regression Loss

↓

Predicted Key Point
```

---

# Common FastAI Components

| Component          | Purpose                    |
| ------------------ | -------------------------- |
| DataBlock          | Defines dataset recipe     |
| ImageBlock         | Input images               |
| MultiCategoryBlock | Multiple labels            |
| get_x              | Locate image               |
| get_y              | Locate labels              |
| splitter           | Train/Validation split     |
| DataLoaders        | Mini-batches               |
| vision_learner     | Build learner              |
| ResNet18           | Pretrained CNN             |
| BCEWithLogitsLoss  | Multi-label loss           |
| accuracy_multi     | Multi-label metric         |
| y_range            | Restrict regression output |

---

# Common Errors

### Tensor on CPU, model on GPU

```
Input type CUDA
Weight type CPU
```

Fix

```python
learn.model.cuda()
```

---

### BCEWithLogitsLoss TypeError

Modern PyTorch versions sometimes require

```python
activs.as_subclass(torch.Tensor)

y.as_subclass(torch.Tensor).float()
```

instead of directly passing fastai tensor subclasses.

---

### plot_function Missing

Newer fastai versions removed `plot_function`.

Use matplotlib instead.

---

# Interview Questions

### What is Multi-label Classification?

Predicting multiple classes for a single image.

---

### Difference between Classification and Regression?

Classification predicts categories.

Regression predicts continuous numbers.

---

### Why MultiCategoryBlock?

Because one image can contain multiple labels.

---

### Why use BCE instead of Cross Entropy?

Because every class is treated independently.

---

### Why sigmoid instead of softmax?

Softmax forces probabilities to compete.

Sigmoid allows multiple classes to be positive simultaneously.

---

### Why y_range?

To constrain regression outputs to the valid target range.

---

# Things Jeremy Wants You to Learn

✔ DataBlock is a recipe.

✔ Multi-label classification predicts multiple classes.

✔ Multi-hot encoding represents multiple labels.

✔ Neural networks output logits, not probabilities.

✔ Sigmoid converts logits into probabilities.

✔ BCE measures prediction error.

✔ Threshold converts probabilities into labels.

✔ Fine-tuning uses pretrained knowledge.

✔ Regression predicts continuous values.

✔ y_range constrains regression outputs.

---

# One-Page Revision

```
Classification
↓

One Label

------------------------

Multi-label

↓

Many Labels

↓

Sigmoid

↓

Binary Cross Entropy

↓

Threshold

------------------------

Regression

↓

Predict Numbers

↓

Coordinates

↓

y_range(-1,1)

------------------------

DataBlock

↓

DataLoaders

↓

ResNet

↓

Prediction

↓

Loss

↓

Backpropagation
```

---

# Key Takeaways

* **Classification:** Predict one class.
* **Multi-label Classification:** Predict multiple independent classes using sigmoid and Binary Cross Entropy.
* **Regression:** Predict continuous numerical values such as coordinates.
* **DataBlock** defines how data is loaded and processed.
* **Transfer learning** allows us to reuse pretrained models and fine-tune them efficiently.
* The same deep learning workflow (Data → Model → Loss → Optimization) applies across all tasks; only the **target type, output layer, and loss function** change.
