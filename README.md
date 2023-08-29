# NeuralNetworkWithKerasTuner
# Hyperparameter Tuning for Neural Network Model

This repository contains a Python script for building and tuning a neural network model using Keras Tuner to predict a target variable. The code is designed to optimize the architecture and hyperparameters of the neural network for the best performance on a given dataset.

## Description

The code demonstrates the process of building and tuning a neural network model using Keras Tuner. The model is designed to predict a target variable based on input features. It iterates through different architectures and hyperparameters to find the optimal configuration that minimizes the validation loss.

### Neural Network Architecture

The neural network architecture is defined as follows:

- Input Layer: The input shape is determined by the shape of the training data.
- Hidden Layers: The number of hidden layers and their units are tuned using the Keras Tuner. The activation functions can be either 'selu' or 'relu'.
- Output Layer: A single output neuron is used for regression tasks.

### Mathematical Formulas
- The model architecture can be represented as:
  # $\[\text{{Model}}(x) = \text{{Input Layer}} \rightarrow \text{{Hidden Layers}} \rightarrow \text{{Output Layer}}\]$
- The hidden layer computation for layer $\(i\)$ is given by:
  $\[
  h_i = \text{{Activation}}(\text{{Input}} \cdot \text{{Weights}}_i + \text{{Bias}}_i)
  \]$

### Hyperparameters

The following hyperparameters are tuned using Keras Tuner:

- Number of hidden layers: An integer between 2 and 5.
- Units in each hidden layer: An integer between 64 and 512.
- Activation function: Either 'selu' or 'relu'.
- Learning rate: A float between 1e-4 and 1e-2 (log scale).

### Hyperparameter Tuning

The code uses RandomSearch from Keras Tuner to search the hyperparameter space for the best configuration. The objective is to minimize the validation loss. A maximum of 256 trials is performed.

### Results

After tuning, the best model is selected based on the lowest validation loss. The summary of the best model's architecture and hyperparameters is displayed.
- The mean squared error (MSE) loss function is defined as:
  ### $$\[\text{{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2\]$$
- The mean absolute error (MAE) metric is defined as:
  ### $$\[\text{{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|\]$$

## Formulae

The mathematical formulas corresponding to the architecture and hyperparameters are as follows:

- Number of Hidden Layers: $\( num\_layers \)$
- Units in Hidden Layer $\( i \): \( units_i \)$
- Activation Function: $\( activation \)$
- Learning Rate: $\( lr \)$
