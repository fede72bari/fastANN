# Changelog

All notable changes to `fastANN` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.3.0] - 2026-10-04

### Added
- `backend = 'jax'`: the network runs on JAX too (the backend to use on TPUs); `compute_gradients` /
  `gradient_feature_importance` support it. Models are portable across TensorFlow, PyTorch and JAX.
- `sample_weight`: one weight per row of `X_data` (or of `X_train_s` with pre-split inputs), passed to `fit` for the
  training rows.
- `monitor_auc` and `monitor_auc_rows`: ROC AUC of the test-set predictions computed at the end of every epoch,
  optionally on a subset of rows, logged as `val_monitored_auc` and usable to choose the epoch with early stopping
  and checkpoint (mode `'max'`). Module function `make_auc_callback()` and method `auc_callbacks()`.
- `hidden_layer_type = 'gated_fan'` (with `periodic_share`, `gated`, `frequency_init_std`): gated Fourier Analysis
  Network hidden layers instead of `Dense`. Module function `fan_layers()`; the layer is the one of
  `fastGatedFourierAnalysisNetwork` and saved models load in both packages.
- The new settings are saved in the hyperparameters JSON (the weight and row arrays only as flags).

### Changed
- The random split now splits row positions with the same seed (identical rows to the previous versions), so that
  weights and row masks follow the rows.

## [2.2.0] - 2026-10-02

### Added
- `network_training(..., callbacks=None)`: extra Keras callbacks (e.g. a time
  limit or a learning-rate schedule) run together with early stopping and the
  best-epoch checkpoint.

## [2.1.0] - 2026-10-02

### Added
- `shuffle` parameter (default `True`, the Keras default used so far), passed
  to `fit`: permutes the training rows among the batches at every epoch. Same
  name as in `fastLSTM`. Saved in the hyperparameters file (older files load
  as `True`).

## [2.0.1] - 2026-10-02

### Fixed
- `metrics=['accuracy']` on a model with several sigmoid outputs (several binary targets) was resolved by Keras 3 to
  `CategoricalAccuracy`, which compares only the arg-max of the outputs: on
  sigmoid outputs this reports meaningless, much too high values (a constant
  model scored about 0.92 instead of about 0.47) and misled early stopping and
  checkpointing on `val_accuracy`. The new `compile_metrics()` method maps
  `'accuracy'`/`'acc'` to `BinaryAccuracy` with sigmoid and to
  `CategoricalAccuracy` with softmax, keeping the logged name `accuracy`.
  Applied both when compiling and when loading a model.

## [2.0.0] - 2026-10-02

Major release: TensorFlow or PyTorch backend, API conventions shared with the
sister package `fastLSTM`, several bug fixes and complete documentation. The
public API is unchanged and models saved by 1.x still load; see "Changed" for
behaviour differences.

### Added
- `backend` parameter: the network runs on Keras 3 with TensorFlow (`'tensorflow'`) or PyTorch (`'torch'`). TensorFlow is no longer required with PyTorch. Saved models reload with either backend.
- `load_keras()` module function.
- Hyperparameters JSON now also stores `Y_scaler_file_name` and `backend`.
- Clear errors for a network that is not built or for `model_dropout` / `model_relative_width` of different lengths.
- `model_summary` now holds the text summary of the network.
- Complete English docstrings for every function, a user manual in the README, this changelog and `fastANN.__version__`.

### Changed
- `split_type = 'random'` now uses `train_size_rate` as the training fraction, as the sequential split does (see Fixed): random-split models trained with 1.x used 30% of the data for training with the default 0.7.
- `load_all()` reads every file from the folder of the JSON instead of the `data_storage_path` stored inside it.
- Loaded models are recompiled with the saved loss, metrics and learning rate (fresh optimizer state).
- The `ModelCheckpoint` callback is stored in `self.model_checkpoint`.
- `early_stop_patience_set(patience=None)`, `checkpoint_callback(save_best_only=None)`, `load_model(model_file_name=None, ...)`, `load_training_history(training_history_file_name=None, ...)` and `gradient_feature_importance(feature_names=None)` have optional arguments, as in fastLSTM.
- The network starts with an explicit `Input` layer.
- Gradient feature importance uses the targets the network was trained on (scaled targets, or the features in autoencoder mode).
- Keras/TensorFlow is no longer imported when the module is imported.

### Fixed
- `split_type = 'random'` used `train_size_rate` as the TEST fraction.
- A second `network_training()` on the same object crashed (the checkpoint callback overwrote the method).
- `binary_precision_recall_vs_scoring()` always failed (it unpacked 2 of the 3 returned values).
- `load_scaler(Y_scaler_file_name=...)` raised `NameError`.
- `fastANN()` without data crashed, so `fastANN().load_all(...)` could not be used; pre-split inputs and `save_X_Y_data` without `X_data` now work.
- `load_all()` could not load a model folder that had been moved or copied.
- Import failed with Plotly ≥ 7 (`create_candlestick`).

## [1.0.0] - 2026-05-12

- Dense network classifier/regressor/autoencoder with sequential or random split, scaling, early stopping, checkpointing and saving/loading of model, scalers, history, hyperparameters and data.
