# fastANN

**A structured framework to build, train, version and reuse dense (feed-forward) neural networks, with a few lines of code.**

## Overview: why fastANN

Training a neural network "by hand" with Keras means writing, every time, the same plumbing: splitting the data, fitting scalers only on the training set, sizing the input and output layers, adding early stopping and checkpoints, and then saving the model **together with** the scaler, the data and the settings that produced it. That plumbing is where many bugs hide (a scaler fitted on the test set, a model reloaded with the wrong scaler, a "best" model nobody can reproduce) and where much of the experiment time goes.

`fastANN` turns it into a tested, reusable class, so a data scientist can focus on the questions that matter: *which features, which targets, which architecture*.

What it does for you:

- **Correct data preparation** — sequential (time series) or random train/test split, scalers fitted on the training set only, optional target scaling with automatic descaling of predictions.
- **Architecture from a short description** — `model_relative_width = [2, 1]` means "two hidden layers, twice and once the number of features"; input and output layers are sized automatically.
- **Three kinds of models with the same code** — classifiers, regressors and **autoencoders** (`autoencoder_mode = True`), with any Keras activation or a trainable `'PReLU'`.
- **Training best practices built in** — early stopping, checkpoint of the best epoch (reloaded at the end), training-history plots.
- **Reproducibility and model versioning** — every training run is saved as a self-describing, timestamped set of files (model, scalers, data, hyperparameters, history) and restored with one call (see [Model versioning](#model-versioning-runs-datasets-and-hyperparameters)).
- **Evaluation and use** — classification reports, precision/recall vs probability cutoff, gradient-based feature importance, prediction on new data with automatic scaling/descaling.
- **TensorFlow or PyTorch** — choose the framework when creating the instance (`backend = 'tensorflow'` or `'torch'`); the same code, files and results work with both, and a model trained with one backend can be reloaded with the other.
- **One workflow for two model families** — `fastANN` shares parameter names, method names and saved-file layout with its sister package [`fastLSTM`](https://github.com/fede72bari/fastLSTM) (recurrent networks for time series): the same code pattern trains, saves and reloads both, so they can be compared on the same data.

Typical uses: tabular classification (e.g. trade/no-trade signals), regression of continuous targets, anomaly detection and feature compression with autoencoders, quick baselines before moving to sequence models.

---

**Current version: 2.2.0** (`fastANN.__version__`) — see the [CHANGELOG](CHANGELOG.md).

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Key concepts](#key-concepts)
4. [Constructor parameters](#constructor-parameters)
5. [Methods reference](#methods-reference)
6. [Saved files](#saved-files)
7. [Model versioning: runs, datasets and hyperparameters](#model-versioning-runs-datasets-and-hyperparameters)
8. [Examples](#examples)
9. [Differences from fastLSTM](#differences-from-fastlstm)
10. [Tips and caveats](#tips-and-caveats)

---

## Installation

The package is the `fastANN` folder of this repository (no `pip` package yet).

```bash
git clone https://github.com/fede72bari/fastANN.git
```

Then either copy the inner `fastANN/` folder next to your notebook/script, or add the repository folder to the Python path:

```python
import sys
sys.path.append('/path/to/fastANN')   # the cloned repository folder
from fastANN import fastANN
```

### Requirements

Python 3.9+ and:

| Purpose | Packages |
|---|---|
| Core | `keras` 3 with **one** backend: `tensorflow` (2.16+) **or** `torch`; `scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib` |
| Plots and notebooks | `matplotlib`, `plotly`, `ipython` |
| Imported by the module (shared toolbox) | `xgboost`, `seaborn`, `tabulate`, `statsmodels`, `imbalanced-learn`, `deap`, `yfinance`, `pytz` |

```bash
pip install scikit-learn pandas numpy scipy joblib matplotlib plotly ipython \
            xgboost seaborn tabulate statsmodels imbalanced-learn deap yfinance pytz
pip install tensorflow          # TensorFlow backend (includes Keras 3)
pip install keras torch         # PyTorch backend (TensorFlow is then optional)
```

Tested with Keras 3.15 on TensorFlow 2.21 and on PyTorch 2.14, pandas 3.0, NumPy 2.4, scikit-learn 1.9, SciPy 1.17 and Plotly 7.

---

## Quick start

```python
from fastANN import fastANN

# X_df: features, one row per sample;  Y_df: targets (here a 0/1 column 'signal')
ann = fastANN(X_data = X_df,
              Y_data = Y_df[['signal']],
              model_relative_width = [2, 1],      # two hidden layers
              model_dropout = [0.2, 0.1],
              data_storage_path = './models/',    # must end with a separator
              model_name = 'signal')

ann.network_structure_set_compile()                 # build + compile
ann.network_training(epochs = 200, batch_size = 64) # train, save everything, reload best epoch

results_df, probabilities_df = ann.network_predictions_evaluation(min_probability = 0.5)
```

Later, in another session:

```python
ann = fastANN()
ann.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF ANN MODEL - signal.json',
             file_path_name = './models/')
predictions = ann.model_predict(new_X_df)           # scaled inside, descaled if needed
```

---

## Key concepts

### Network architecture

`model_relative_width` lists the **hidden** layers only; the input and output layers are always added automatically:

```
Input(n_features)
Dense(n_features * model_relative_width[0])  [+ PReLU] + Dropout(model_dropout[0])
...
Dense(n_features * model_relative_width[-1]) [+ PReLU] + Dropout(model_dropout[-1])
Dense(n_targets, last_layer_activation)                 # n_features in autoencoder mode
```

- Widths are **relative to the number of features**: with 10 features, `[2, 1]` gives layers of 20 and 10 units.
- `model_dropout` must have the same length as `model_relative_width`.
- With `activation = 'PReLU'` each hidden Dense layer is linear and followed by a trainable PReLU layer.

### Choosing TensorFlow or PyTorch

The network is written with Keras 3, which runs on top of TensorFlow or PyTorch. Choose the backend when you create the first instance:

```python
model = fastANN(X_data = X_df, Y_data = Y_df, backend = 'torch')       # or 'tensorflow'
print(model.backend)                                               # 'torch'
```

- `backend = None` (default) uses the backend already active, or the `KERAS_BACKEND` environment variable, or `'tensorflow'`.
- Keras uses **one backend per Python process**: the first instance fixes it. Asking for a different one later raises a clear error; restart the kernel to switch.
- Saved models are portable: a model trained with TensorFlow can be reloaded with PyTorch and vice versa (`load_all` reloads the weights and recompiles the network with the saved loss, metrics and learning rate).
- With PyTorch, create the first instance (or `import torch`) **before** anything that imports TensorFlow: with some TensorFlow/PyTorch builds, loading the Keras PyTorch backend after TensorFlow crashes Python.
- `model.keras` is the Keras module in use; `model.model` is a regular Keras model on either backend.

### Split and scaling

- `split_type = 'sequential'` (default): the first `train_size_rate` of the rows is the training set — use it for time series to avoid look-ahead.
- `split_type = 'random'`: shuffled split (seed 42) with the same training fraction.
- Features are scaled with a `StandardScaler` fitted on the training set only. Targets are scaled only with `scale_targets = True`; `model_predict` then returns predictions in the original scale.
- Already split and scaled arrays can be passed directly (`X_train_s`, `Y_train`, `X_test_s`, `Y_test`) when `X_data`/`Y_data` are not given.

### Model types

| Model | Settings |
|---|---|
| Binary / multi-label classifier | `last_layer_activation = 'sigmoid'`, `loss = 'binary_crossentropy'`, monitor `'val_accuracy'` mode `'max'` |
| Multi-class classifier | one-hot targets, `last_layer_activation = 'softmax'`, `loss = 'categorical_crossentropy'` |
| Regressor | `last_layer_activation = 'linear'`, `loss = 'mse'`, monitor `'val_loss'` mode `'min'`, optionally `scale_targets = True` |
| Autoencoder | `autoencoder_mode = True` (targets = scaled features, output size = `n_features`), `last_layer_activation = 'linear'`, `loss = 'mse'` |

---

## Constructor parameters

`fastANN(**parameters)` — all parameters are optional keywords.

### Data

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `X_data` | `DataFrame` | `None` | Features, one row per sample. Required to train (unless pre-split data are given); omit it when restoring with `load_all()`. |
| `Y_data` | `DataFrame` | `None` | Targets aligned with `X_data` (one column per target; 0/1 for binary classification). |
| `X_train_s`, `X_test_s` | array | `None` | Already split **and scaled** features, used only without `X_data`/`Y_data`. |
| `Y_train`, `Y_test` | `DataFrame` | `None` | Already split targets, used together with `X_train_s`/`X_test_s`. |
| `split_type` | `'sequential'`, `'random'` | `'sequential'` | Train/test split method. |
| `shuffle` | `bool` | `True` | Permute the training rows among the batches at every epoch (Keras `fit(shuffle=...)`). Changes only the presentation order, never the train/test split. Same name as in fastLSTM. |
| `train_size_rate` | `float` 0–1 | `0.7` | Fraction of rows used for training. |
| `scale_targets` | `bool` | `False` | Scale the targets too; predictions are descaled by `model_predict`. |
| `save_X_Y_data` | `bool` | `True` | Save `X_data`/`Y_data` as CSV at training time, so `load_all()` can rebuild the same split. |

### Architecture

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `model_relative_width` | `list` of `float` | `[1]` | Width of each hidden layer relative to the number of features. Its length = number of hidden layers. |
| `model_dropout` | `list` of `float` 0–1 | `[0]` | Dropout after each hidden layer (same length as `model_relative_width`). |
| `activation` | Keras activation name or `'PReLU'` | `'relu'` | Activation of the hidden layers. |
| `last_layer_activation` | `'sigmoid'`, `'softmax'`, `'linear'`, … | `'sigmoid'` | Activation of the output layer. |
| `autoencoder_mode` | `bool` | `False` | Train the network to reproduce its scaled input. |

### Training

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `0.0003` | Adam learning rate. |
| `loss` | Keras loss name or object | `'binary_crossentropy'` | e.g. `'mse'`, `'mae'`, `'categorical_crossentropy'`. |
| `metrics` | `list` of `str` | `['accuracy']` | Metrics logged by Keras (validation ones get the `val_` prefix). |
| `history_metrics` | `list` of `str` | `['accuracy', 'val_accuracy']` | Columns plotted by `plot_training_history()` (use `['loss', 'val_loss']` for regressors/autoencoders). |

The batch size is given to `network_training(epochs, batch_size)`.

### Early stopping and checkpoint

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `early_stop_monitor_metric` | `str` | `'val_accuracy'` | Quantity monitored by early stopping. |
| `early_stop_mode` | `'max'`, `'min'`, `'auto'` | `'max'` | Whether it must increase or decrease. |
| `early_stop_patience` | `int` | `200` | Epochs without improvement before stopping. |
| `checkpoint_monitor_metric` | `str` | `'val_accuracy'` | Quantity used to pick the best epoch to save and reload. |
| `checkpoint_mode` | `'max'`, `'min'`, `'auto'` | `'max'` | Whether it must increase or decrease. |
| `save_best_only` | `bool` | `True` | Save only improving epochs (otherwise the last epoch is kept). |

### Storage

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_storage_path` | `str` | `'\\cyPredict\\'` | Folder for every saved file; it is concatenated to file names, so it **must end with a separator** (`'./models/'`). |
| `model_name` | `str` | `'ANN'` | Name used in every saved file name. |
| `backend` | `'tensorflow'`, `'torch'` | `None` | Framework that runs the network (see [Choosing TensorFlow or PyTorch](#choosing-tensorflow-or-pytorch)). |

---

## Methods reference

Every method has a complete docstring: `help(fastANN.network_training)`.

### Build and train

| Method | Description |
|---|---|
| `network_structure_set_compile()` | Builds the network (see [architecture](#network-architecture)) and compiles it with Adam, `loss` and `metrics`. The text summary is kept in `model_summary`. |
| `network_training(epochs, batch_size, callbacks=None)` | Trains with early stopping and checkpointing, saves every artefact (see [Saved files](#saved-files)), reloads the best epoch into `model` and plots the history. (extra Keras `callbacks` optional) |
| `split_and_scale(scaler_fit=False)` | Splits `X_data`/`Y_data` (by `split_type`) and scales them. `scaler_fit=True` fits the scalers (new data), `False` only applies them. Called by the constructor with `True`. |
| `early_stop_patience_set(patience=None)` | Rebuilds the early stopping callback, optionally with a new patience. |
| `checkpoint_callback(save_best_only=None)` | Creates the `ModelCheckpoint` callback (`model_checkpoint`). Called by `network_training`. |

### Evaluate

| Method | Returns | Description |
|---|---|---|
| `network_predictions_evaluation(min_probability, output_dict=False)` | `(results_df, probabilities_df)` or `(results_df, probabilities_df, report)` | Classifiers: test-set predictions thresholded at `min_probability` and compared with the actual values; a `classification_report` is printed for every target column. `report` is the dict of the last column. |
| `binary_precision_recall_vs_scoring(n_points=15, plot=True)` | `DataFrame` (`Cutoff`, `Precision`, `Recall`) | Precision and recall of class `1` for cutoffs from `n_points/100` to `0.99` (Plotly chart). Labels must be integers 0/1. |
| `plot_training_history()` | – | Plots the `history_metrics` columns of `loss_df`. |
| `gradient_feature_importance(feature_names=None)` | `(importances, names)` sorted increasingly | Mean absolute gradient of the loss w.r.t. each input feature on the test set (Plotly bar chart). |
| `compute_gradients(inputs, targets)` | tensor | Gradient of the MSE w.r.t. the inputs (used by the method above). |

### Predict

| Method | Returns | Description |
|---|---|---|
| `model_predict(data, apply_scaler=True, descale_result=True)` | array `(n_samples, n_outputs)` | Predicts on new samples (same columns as `X_data`). Scales the inputs and descales the outputs (if `scale_targets`). |

### Save and load

| Method | Description |
|---|---|
| `load_all(hyperparameters_file_name=None, file_path_name=None)` | Restores everything: hyperparameters, model, scalers, training history and (if saved) data, split and scaled with the loaded scalers. Files are read from `file_path_name` (the folder of the JSON), so model folders can be moved. |
| `load_hyperparameters(file_name, file_path_name=None)` | Reads the JSON into `hyperparameters`. |
| `set_hyperparameters()` | Applies `hyperparameters` to the attributes. |
| `load_model(model_file_name=None, file_path_name=None)` | Loads the `.keras` model. |
| `load_scaler(scaler_file_name=None, Y_scaler_file_name=None, file_path_name=None)` | Loads the scalers. |
| `load_training_history(training_history_file_name=None, file_path_name=None)` | Loads the history CSV into `loss_df`. |
| `save_hyperparameters(file_name)` / `init_hyperparameters(...)` | Write / rebuild the hyperparameters dictionary (called by `network_training`). |

### Main attributes

| Attribute | Content |
|---|---|
| `model` | The Keras model. |
| `scaler`, `Y_scaler` | Features and targets scalers (`Y_scaler` is `None` unless `scale_targets`). |
| `X_train`, `Y_train`, `X_test`, `Y_test` | Unscaled split (DataFrames). |
| `X_train_s`, `X_test_s`, `Y_train_s`, `Y_test_s` | Data used for training. |
| `loss_df` | Training history (one row per epoch). |
| `hyperparameters` | Dictionary saved as JSON. |
| `model_summary` | Text summary of the network. |

---

## Saved files

`network_training()` writes into `data_storage_path` (`<dt>` = training timestamp, `<name>` = `model_name`):

| File | Content |
|---|---|
| `<dt> - ANN MODEL - <name>.keras` | Best (or last) model. |
| `<dt> - SCALER FOR ANN MODEL - <name>.pkl` | Features scaler. |
| `<dt> - Y SCALER FOR ANN MODEL - <name>.pkl` | Targets scaler (only with `scale_targets`). |
| `<dt> - TRAINING HISTORY OF ANN MODEL - <name>.csv` | Loss and metrics per epoch. |
| `<dt> - HYPERPARAMETERS OF ANN MODEL - <name>.json` | All settings and the names of the other files. |
| `<dt> - X_data FOR ANN MODEL - <name>.csv`, `<dt> - Y_data FOR ANN MODEL - <name>.csv` | Data (only with `save_X_Y_data`; the index is not saved). |

To restore a run you only need the JSON name and its folder: `fastANN().load_all(json_name, file_path_name = folder)`.

---

## Model versioning: runs, datasets and hyperparameters

Every call to `network_training()` is a **run**, identified by its timestamp. A run writes a complete, self-describing snapshot: the JSON contains all hyperparameters *and* the names of the model, scaler, history and data files of that same run, so each model version stays linked to the exact dataset and settings that produced it.

```
models/
├── 2025-03-07 10-00-00 - HYPERPARAMETERS OF ANN MODEL - signal.json   ← entry point of the run
├── 2025-03-07 10-00-00 - ANN MODEL - signal.keras
├── 2025-03-07 10-00-00 - SCALER FOR ANN MODEL - signal.pkl
├── 2025-03-07 10-00-00 - TRAINING HISTORY OF ANN MODEL - signal.csv
├── 2025-03-07 10-00-00 - X_data FOR ANN MODEL - signal.csv
├── 2025-03-07 10-00-00 - Y_data FOR ANN MODEL - signal.csv
├── 2025-03-08 15-30-12 - HYPERPARAMETERS OF ANN MODEL - signal.json   ← a later run, same model name
└── ...
```

What the JSON records: architecture (`model_relative_width`, `model_dropout`, `activation`, `autoencoder_mode`, …), training settings (`loss`, `metrics`, `learning_rate`, `batch_size`, early stopping and checkpoint settings), data settings (`split_type`, `shuffle`, `train_size_rate`, `scale_targets`, feature and target column names) and the file names of the run.

### Recommended practices

1. **Keep `save_X_Y_data = True`** (default): the exact training data are saved with the model, so `load_all()` rebuilds the same split and you can always re-evaluate or audit a version.
2. **Use `model_name` for the experiment, the timestamp for the version**: e.g. `model_name = 'signal_v2_20feat'`; every retraining adds a new timestamped run without overwriting the previous ones.
3. **One folder per project** (`data_storage_path`), and move or copy whole folders freely: `load_all(json, file_path_name = new_folder)` reads every file from the folder of the JSON.
4. **Compare versions** by reading their JSON and history files:

```python
import glob, json, os
import pandas as pd

rows = []
for path in glob.glob('./models/* - HYPERPARAMETERS OF ANN MODEL - signal*.json'):
    hp = json.load(open(path))
    history = pd.read_csv('./models/' + hp['training_history_file_name'], index_col = 0)
    rows.append({'run': hp['model_training_datetime'],
                 'layers': hp['model_relative_width'],
                 'activation': hp['activation'],
                 'best_val_accuracy': history['val_accuracy'].max(),
                 'json': os.path.basename(path)})
runs = pd.DataFrame(rows).sort_values('best_val_accuracy', ascending = False)
```

5. **Reload any version** with its JSON name: `fastANN().load_all(runs.iloc[0]['json'], file_path_name = './models/')`.
6. **Remember which file to use in production**: the JSON name is the only reference you need to store (in a config file, a database, a Git tag…).
7. For long-term traceability you can version the folder itself (Git LFS, DVC, cloud storage); the file names already carry timestamp and experiment name.

---

## Examples

### 1. Binary classification and cutoff analysis

```python
ann = fastANN(X_data = X_df, Y_data = Y_df[['signal']],
              model_relative_width = [2, 1], model_dropout = [0.2, 0.1],
              early_stop_patience = 30,
              data_storage_path = './models/', model_name = 'signal')
ann.network_structure_set_compile()
ann.network_training(epochs = 300, batch_size = 64)

results_df, probs_df, report = ann.network_predictions_evaluation(0.5, output_dict = True)
print(report['1']['precision'], report['1']['recall'])

pr_df = ann.binary_precision_recall_vs_scoring(n_points = 30)   # cutoffs 0.30 ... 0.99
best = pr_df.loc[pr_df['Precision'].idxmax()]
```

### 2. Regression with scaled targets and random split

```python
ann = fastANN(X_data = X_df, Y_data = Y_df[['price']],
              split_type = 'random',
              last_layer_activation = 'linear', loss = 'mse', metrics = ['mae'],
              history_metrics = ['loss', 'val_loss'],
              early_stop_monitor_metric = 'val_loss', early_stop_mode = 'min',
              checkpoint_monitor_metric = 'val_loss', checkpoint_mode = 'min',
              scale_targets = True,
              data_storage_path = './models/', model_name = 'price')
ann.network_structure_set_compile()
ann.network_training(epochs = 200, batch_size = 32)
predicted_prices = ann.model_predict(new_X_df)      # original scale
```

### 3. Autoencoder with PReLU (feature compression / anomaly detection)

```python
ae = fastANN(X_data = X_df, Y_data = X_df, autoencoder_mode = True,
             activation = 'PReLU',
             model_relative_width = [0.5, 0.25, 0.5], model_dropout = [0, 0, 0],
             last_layer_activation = 'linear', loss = 'mse', metrics = ['mae'],
             history_metrics = ['loss', 'val_loss'],
             early_stop_monitor_metric = 'val_loss', early_stop_mode = 'min',
             checkpoint_monitor_metric = 'val_loss', checkpoint_mode = 'min',
             data_storage_path = './models/', model_name = 'autoencoder')
ae.network_structure_set_compile()
ae.network_training(epochs = 300, batch_size = 64)

import numpy as np
reconstruction = ae.model.predict(ae.X_test_s)
errors = np.mean((reconstruction - ae.X_test_s) ** 2, axis = 1)   # high error = anomaly
```

### 4. Pre-split data

```python
ann = fastANN(X_train_s = X_train_scaled, Y_train = Y_train_df,
              X_test_s = X_test_scaled, Y_test = Y_test_df,
              data_storage_path = './models/', model_name = 'presplit')
ann.network_structure_set_compile()
ann.network_training(epochs = 100, batch_size = 32)
```

### 5. Restore a model and evaluate it on new data

```python
ann = fastANN()
ann.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF ANN MODEL - signal.json',
             file_path_name = './models/')

ann.X_data, ann.Y_data = new_X_df, new_Y_df[['signal']]
ann.split_and_scale(scaler_fit = False)           # keep the scaler fitted at training time
ann.network_predictions_evaluation(0.5)
```

### 6. Feature importance

```python
importances, names = ann.gradient_feature_importance()
print(names[-5:])   # five most important features
```

---

## Differences from fastLSTM

Same parameter names, method names and saved-file layout. fastANN-only: `split_type`, `autoencoder_mode`, pre-split inputs (`X_train_s`, `Y_train`, `X_test_s`, `Y_test`) and the `'PReLU'` activation. fastLSTM-only: `LSTM_type`, `timesteps`, `steps_ahead` (multi-step forecasting), `class_weight`, `scaler_type`, sequence helpers (`create_generators`, `prepare_input_sample`, `output_column_names`).

---

## Tips and caveats

- `data_storage_path` must end with `/` (or `\\` on Windows) and the folder must exist.
- For time series keep `split_type = 'sequential'`: a random split leaks future information into training.
- Monitor validation metrics (`val_…`) for both early stopping and checkpoint, and set the modes coherently (`'max'` for accuracy, `'min'` for losses); for regressors and autoencoders also set `history_metrics = ['loss', 'val_loss']`.
- `binary_precision_recall_vs_scoring` and the `report` returned by `network_predictions_evaluation` refer to the **last** target column.
- Inside Jupyter the plots appear inline; in scripts call `matplotlib.pyplot.show()` after `plot_training_history()`.
