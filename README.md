# fastANN: A Lightweight Artificial Neural Network Framework

## Overview

### What is fastANN?
`fastANN` is a lightweight and flexible artificial neural network (ANN) framework designed to simplify the process of training deep learning models. Unlike building models manually using TensorFlow or Keras, `fastANN` abstracts away much of the complexity, providing an easy-to-use API for rapid experimentation with various model architectures, training parameters, and dataset configurations.

### Why Use fastANN?
- **Simplifies Model Creation**: Eliminates the need for manually defining layers, optimizers, and loss functions.
- **Pre-configured Training Flow**: Handles data scaling, early stopping, and checkpointing automatically.
- **Supports Autoencoders**: Includes built-in support for autoencoder training.
- **Flexible Data Handling**: Accepts raw datasets and automatically handles splitting, scaling, and training processes.

---

## Hyperparameters

### Model Architecture Parameters
| Parameter                  | Description |
|----------------------------|-------------|
| `model_relative_width`     | Defines the relative size of each hidden layer compared to the input size. |
| `model_dropout`            | List of dropout values for each layer to prevent overfitting. |
| `activation`               | Activation function for the hidden layers (e.g., 'relu', 'tanh', 'PReLU'). |
| `last_layer_activation`    | Activation function for the output layer (e.g., 'sigmoid', 'softmax'). |

### Training Parameters
| Parameter                  | Description |
|----------------------------|-------------|
| `learning_rate`            | Learning rate for the optimizer. Default is `0.0003`. |
| `loss`                     | Loss function (e.g., 'binary_crossentropy'). |
| `metrics`                  | List of metrics to track during training (e.g., `['accuracy']`). |
| `train_size_rate`          | Percentage of data to use for training (default: `0.7`). |
| `split_type`               | Splitting method: `'random'` or `'sequential'`. |

### Early Stopping and Checkpoints
| Parameter                  | Description |
|----------------------------|-------------|
| `early_stop_monitor_metric` | Metric to monitor for early stopping. Default: `'val_accuracy'`. |
| `checkpoint_monitor_metric` | Metric for saving the best model checkpoint. |
| `early_stop_mode`           | `'max'` or `'min'` (depends on metric type). |
| `early_stop_patience`       | Number of epochs to wait before stopping if no improvement is detected. |
| `save_best_only`            | If `True`, only saves the best model during training. |

---

## Model and Data Saving Mechanisms

fastANN includes built-in functionalities for saving models, training data, scalers, and hyperparameters to ensure full reproducibility and ease of use.

### **Model Saving**
- During training, the best-performing model (based on the checkpoint monitor metric) is saved automatically.
- The model file is stored in the specified `data_storage_path` with a timestamped filename in `.keras` format.
- Example filename: `2025-03-07 - ANN MODEL - fastANN.keras`

### **Data Saving**
- If `save_X_Y_data=True`, the training dataset (`X_data` and `Y_data`) is saved as `.csv` files in the `data_storage_path`.
- Example filenames:
  - `2025-03-07 - X_data FOR ANN MODEL - fastANN.csv`
  - `2025-03-07 - Y_data FOR ANN MODEL - fastANN.csv`

### **Scaler Saving**
- The scaler used for feature normalization is saved as a `.pkl` file.
- If `scale_targets=True`, an additional target scaler is saved.
- Example filenames:
  - `2025-03-07 - SCALER FOR ANN MODEL - fastANN.pkl`
  - `2025-03-07 - Y SCALER FOR ANN MODEL - fastANN.pkl`

### **Training History Saving**
- The training history, including loss and accuracy metrics over epochs, is stored in a `.csv` file.
- Example filename:
  - `2025-03-07 - TRAINING HISTORY OF ANN MODEL - fastANN.csv`

### **Hyperparameters Saving**
- Hyperparameters are stored as a `.json` file for later reuse.
- Example filename:
  - `2025-03-07 - HYPERPARAMETERS OF ANN MODEL - fastANN.json`

### **Loading Saved Models and Data**
To restore a previously trained model with all its settings:
```python
model.load_all("hyperparameters.json")
```
This function loads the model, scaler, training history, and dataset split, ensuring the same conditions as the original training session.

---

## Function Parameters and Inputs

### `network_structure_set_compile()`
Configures the ANN structure and compiles the model.
#### **Parameters:**
- None (uses attributes set in `__init__`).

### `network_training(epochs, batch_size)`
Trains the ANN with the given dataset.
#### **Parameters:**
- `epochs` (int): Number of training iterations.
- `batch_size` (int): Size of each batch used in training.

### `model_predict(data, apply_scaler=True, descale_result=True)`
Generates predictions for input data.
#### **Parameters:**
- `data` (array-like): Input features for prediction.
- `apply_scaler` (bool): Whether to apply scaling before prediction (default: `True`).
- `descale_result` (bool): If `True`, reverses scaling on output predictions.

### `network_predictions_evaluation(min_probability, output_dict=False)`
Evaluates model performance on test data.
#### **Parameters:**
- `min_probability` (float): Probability threshold for classification.
- `output_dict` (bool): If `True`, returns detailed classification metrics.

### `plot_training_history()`
Plots the training history of loss and accuracy metrics over epochs.
#### **Parameters:**
- None (uses training history stored during model training).

### `load_all(hyperparameters_file_name)`
Loads a previously saved model, hyperparameters, and dataset splits.
#### **Parameters:**
- `hyperparameters_file_name` (str): Path to the JSON file storing hyperparameters.

---

## Conclusion
`fastANN` is designed to streamline the workflow of training and experimenting with neural networks. By automating key processes such as data scaling, early stopping, and model checkpointing, it allows users to focus on model performance and dataset exploration rather than boilerplate code. Happy coding! 🚀

