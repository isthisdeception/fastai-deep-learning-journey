# Chapter 4 Notes - MNIST Basics, Neural Networks, and Learners

## Summary

This chapter taught me how a neural network actually learns.

Before this chapter, I thought training a neural network was some kind of magic hidden behind `learn.fit()`. After working through the concepts and implementing my own Learner, I now understand that training is simply:

1. Make a prediction
2. Calculate the error (loss)
3. Compute gradients
4. Update parameters
5. Repeat

Everything else is an abstraction built on top of this loop.

---

# Key Concepts

## Weight

A weight determines how important an input is.

Example:

```python
prediction = (flat_img * weights).sum()
```

Each pixel has a corresponding weight.

* Large positive weight → pixel strongly increases prediction
* Large negative weight → pixel strongly decreases prediction
* Weight near zero → pixel has little influence

---

## Bias

A bias is a learnable offset added to the final prediction.

```python
prediction = (flat_img * weights).sum() + bias
```

My explanation:

> Weights decide how much each input matters. Bias shifts the final prediction up or down regardless of the inputs.

---

## Prediction

The model's guess.

Example:

```python
prediction = x_train @ weights + bias
```

The goal of training is to make predictions closer to the correct labels.

---

## Loss

Loss measures how wrong the prediction is.

Example:

```python
loss = (prediction - label)**2
```

or

```python
loss = ((prediction - y_train)**2).mean()
```

My explanation:

> Loss is a score that tells the model how bad its predictions are.

Lower loss = better model.

---

## Gradient

Gradients tell us how each parameter affects the loss.

Computed using:

```python
loss.backward()
```

My explanation:

> Gradients tell the model which direction to move each parameter in order to reduce the loss.

---

## Learning Rate

The learning rate controls how big a step we take when updating parameters.

```python
weights -= lr * weights.grad
```

Important lesson:

* Too large → overshoots and training becomes unstable
* Too small → learns very slowly

---

## Gradient Descent

Parameter update rule:

```python
weights -= lr * weights.grad
```

My explanation:

> The model uses gradients as directions and the learning rate as the size of the step.

---

## Why We Need zero_grad()

PyTorch accumulates gradients by default.

Without:

```python
weights.grad.zero_()
```

new gradients are added to old gradients.

My explanation:

> We clear gradients before the next iteration so each update uses only the current gradients.

---

# Matrix Multiplication

Single image:

```python
flat_img.shape
# [784]
```

Many images:

```python
x_train.shape
# [40, 784]
```

Prediction:

```python
prediction = x_train @ weights + bias
```

My explanation:

> Matrix multiplication allows the model to make predictions for many images at once instead of one image at a time.

---

# ReLU

ReLU activation:

```python
hidden = hidden.clamp(min=0)
```

Equivalent to:

```python
ReLU(x) = max(0, x)
```

Example:

```python
[-5, 2, -1, 7]
```

becomes:

```python
[0, 2, 0, 7]
```

My explanation:

> ReLU turns negative neurons off and keeps positive neurons active.

---

# Why ReLU Is Important

Without ReLU:

```text
Linear
↓
Linear
```

can be simplified into:

```text
Linear
```

With ReLU:

```text
Linear
↓
ReLU
↓
Linear
```

the network becomes more powerful and can learn complex patterns.

---

# Neural Network Architecture

The network I built:

```text
784 Inputs
↓
30 Neurons
↓
ReLU
↓
1 Output Neuron
```

Forward pass:

```python
hidden = x_train @ weights1 + bias1
hidden = hidden.clamp(min=0)

output = hidden @ weights2 + bias2
```

---

# Understanding Tensor Shapes

Input:

```python
x_train.shape
# [40, 784]
```

Meaning:

* 40 images
* 784 pixels per image

First layer:

```python
weights1.shape
# [784, 30]
```

Meaning:

* 784 inputs
* 30 neurons

Hidden layer:

```python
hidden.shape
# [40, 30]
```

Meaning:

* 40 images
* 30 outputs per image

Second layer:

```python
weights2.shape
# [30, 1]
```

Output:

```python
output.shape
# [40, 1]
```

Meaning:

* 40 predictions
* 1 prediction per image

---

# Training Loop

The most important idea in the chapter:

```python
for epoch in range(epochs):

    prediction = model(x)

    loss = loss_func(prediction, y)

    loss.backward()

    update_parameters()

    zero_grad()
```

My explanation:

> This loop is the heart of deep learning.

---

# What Is A Learner?

My explanation:

> A Learner is an object that stores data, parameters, and hyperparameters, then runs the training loop through a method like `fit()`.

---

# My Custom Learner

I implemented:

```python
class myLearner:

    def __init__(...):
        ...

    def fit(...):
        ...

    def predict(...):
        ...
```

Features:

* Stores training data
* Stores model parameters
* Trains the neural network
* Makes predictions

---

# Biggest Lessons

1. Neural networks are not magic.
2. Training is just a loop.
3. Gradients tell parameters how to improve.
4. ReLU creates non-linearity.
5. Learner is an abstraction over the training loop.
6. Understanding tensor shapes is critical.
7. Building things from scratch creates deeper understanding than using high-level APIs.

---

# If I Had To Explain Chapter 4 To A Beginner

A neural network learns by:

```text
Input
↓
Prediction
↓
Loss
↓
Backward
↓
Update Parameters
↓
Repeat
```

A Learner simply automates this process.
