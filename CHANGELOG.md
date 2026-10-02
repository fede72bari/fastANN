# Changelog

All notable changes to `fastANN` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
