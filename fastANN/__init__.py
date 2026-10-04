"""
fastANN
=======

A thin, opinionated wrapper around a Keras dense (feed-forward) network for
tabular classification, regression and autoencoding.

The package exposes a single class, :class:`fastANN`, whose public API
(hyperparameter names, method names and saved-file layout) is shared with the
sister package ``fastLSTM``. The network runs on TensorFlow or PyTorch
(Keras 3 backends), chosen with the ``backend`` parameter.

Typical workflow
----------------
>>> from fastANN import fastANN
>>> ann = fastANN(X_data = X_df, Y_data = Y_df,
...               model_relative_width = [2, 1],
...               model_dropout = [0.2, 0.2],
...               data_storage_path = './models/')
>>> ann.network_structure_set_compile()
>>> ann.network_training(epochs = 100, batch_size = 64)
>>> ann.network_predictions_evaluation(min_probability = 0.5)
"""

__version__ = '2.3.0'

# ---------------------------------------------------------------------------
#                              Libraries Import
# ---------------------------------------------------------------------------


# Multiprocessing
import multiprocessing

# Files Management
import sys
import gzip
import joblib
import glob
import csv
import json
import os

# Warnings
import warnings

# Stocks Indicators
# import talib

# Time Management
import datetime
from datetime import datetime, timedelta, date
import time
import pytz
from pytz import timezone


# Math and Sci
import numpy as np
import math
from scipy.signal import argrelextrema
import random
from scipy.signal import find_peaks
from scipy.signal import argrelmax, argrelmin
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simpson as simps
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist


# Reporting
import plotly
try:
    from plotly.figure_factory import create_candlestick
except ImportError:
    # removed from recent Plotly versions; not used by fastANN
    create_candlestick = None
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot
from matplotlib.pylab import rcParams
from xgboost import plot_tree
import seaborn as sns
from tabulate import tabulate
from IPython.display import HTML, display

# Data Management
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


# Machine Learning
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# Keras (TensorFlow or PyTorch backend) is loaded by load_keras, below


# Optimization
from deap import base, creator, tools, algorithms

# Models Explainablity
#import shap
#import torch

# Binary Classification Specific Metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score

# General Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Data sources
import yfinance as yf

# Financial indicators
# import talib



# ---------------------------------------------------------------------------
#                    Deep learning backend (TensorFlow or PyTorch)
# ---------------------------------------------------------------------------

# The network is written with Keras 3, which runs on top of TensorFlow or
# PyTorch. Keras fixes its backend when it is first imported, so it is NOT
# imported here: it is loaded by the first instance created, with the backend
# requested by its `backend` parameter (see load_keras).
SUPPORTED_BACKENDS = ('tensorflow', 'torch')


def load_keras(backend = None):
    """
    Import Keras 3 with the requested backend and return the module.

    Keras uses one backend per Python process: the first call (usually the
    first fastLSTM / fastANN instance) fixes it; later calls must ask for the
    same backend or for ``None``.

    Parameters
    ----------
    backend : {'tensorflow', 'torch'}, optional
        Backend to use. ``None`` keeps the active one, or, if Keras is not
        loaded yet, uses the ``KERAS_BACKEND`` environment variable
        (``'tensorflow'`` when it is not set).

    Returns
    -------
    module
        The ``keras`` module.

    Raises
    ------
    ValueError
        If ``backend`` is not supported.
    RuntimeError
        If Keras is already running in this process with another backend
        (restart the Python kernel to change it).

    Examples
    --------
    >>> keras = load_keras('torch')
    >>> keras.backend.backend()
    'torch'
    """
    if((backend is not None) and (backend not in SUPPORTED_BACKENDS)):
        raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, not '{backend}'.")

    if('keras' not in sys.modules):
        if(backend is not None):
            os.environ['KERAS_BACKEND'] = backend

        if((os.environ.get('KERAS_BACKEND', 'tensorflow') == 'torch') and ('tensorflow' in sys.modules)):
            # with some TensorFlow/PyTorch builds, loading the Keras torch backend after TensorFlow crashes Python
            warnings.warn("TensorFlow is already imported in this process: if Python crashes while loading the "
                          "PyTorch backend, create the first fastLSTM/fastANN instance (or import torch) before "
                          "anything that imports TensorFlow.",
                          RuntimeWarning,
                          stacklevel = 3)

        import keras

    keras = sys.modules['keras']
    active_backend = keras.backend.backend()

    if((backend is not None) and (active_backend != backend)):
        raise RuntimeError(f"Keras is already using the '{active_backend}' backend in this Python process and it "
                           f"cannot be changed safely: restart the kernel to use '{backend}', or create the instance "
                           f"with backend = '{active_backend}' (or None).")

    return keras


# custom layers are registered under the name of the sister package that defines them, so .keras files are
# interchangeable between fastANN and fastGatedFourierAnalysisNetwork
KERAS_PACKAGE = 'fastGatedFourierAnalysisNetwork'


# ---------------------------------------------------------------------------
#                       Gated FAN layer (Keras 3, any backend)
# ---------------------------------------------------------------------------

# The layer classes need the keras module, which is imported only when the first instance is created (the backend
# must be chosen first). fan_layers() defines and registers them once and caches them here.
_FAN_OBJECTS = {}


def periodic_split(units, periodic_share):
    """
    Number of frequencies and of non-periodic units of a gated FAN layer.

    The layer output has ``units`` values: ``2 * d_p`` periodic ones (a cosine
    and a sine for each of the ``d_p`` frequencies) and ``d_p_bar``
    non-periodic ones, with ``2 * d_p`` as close as possible to
    ``units * periodic_share``.

    Parameters
    ----------
    units : int
        Output width of the layer.
    periodic_share : float
        Share (0 - 1) of the output given to the periodic part. 0 gives a
        plain dense layer (no frequencies, no gate).

    Returns
    -------
    d_p : int
        Number of learned frequencies (cosine + sine pairs).
    d_p_bar : int
        Number of non-periodic units.

    Raises
    ------
    ValueError
        If ``periodic_share`` is outside [0, 1) or ``units`` is too small to
        hold both parts.

    Examples
    --------
    >>> periodic_split(368, 1 / 3)
    (61, 246)
    """
    if(not (0 <= periodic_share < 1)):
        raise ValueError(f'periodic_share must be in [0, 1), not {periodic_share}.')

    units = int(units)
    d_p = int(round(units * periodic_share / 2))

    if((periodic_share > 0) and (d_p == 0)):
        d_p = 1

    d_p_bar = units - 2 * d_p

    if(d_p_bar < 1):
        raise ValueError(f'A layer of {units} units cannot hold {2 * d_p} periodic units and at least one '
                         f'non-periodic unit: increase model_relative_width or reduce periodic_share.')

    return d_p, d_p_bar


def fan_layers(keras = None):
    """
    Define (once) and return the custom Keras objects of the package.

    Returns the dictionary ``{'GatedFAN': <layer class>, 'FrequencyClip':
    <constraint class>}``. The classes are registered in the Keras
    serialization registry (package ``'fastGatedFourierAnalysisNetwork'``), so
    saved ``.keras`` files reload with ``keras.models.load_model`` once this
    function has been called; :meth:`fastGatedFourierAnalysisNetwork.load_model`
    does it automatically. The dictionary can also be passed as
    ``custom_objects``.

    Parameters
    ----------
    keras : module, optional
        The ``keras`` module; defaults to :func:`load_keras` ``()``.

    Returns
    -------
    dict
        Custom objects by name.

    Examples
    --------
    >>> objects = fan_layers()
    >>> layer = objects['GatedFAN'](units = 96)
    """
    if(_FAN_OBJECTS):
        return _FAN_OBJECTS

    if(keras is None):
        keras = load_keras()

    ops = keras.ops


    @keras.saving.register_keras_serializable(package = KERAS_PACKAGE)
    class FrequencyClip(keras.constraints.Constraint):
        """
        Keep the absolute value of every frequency inside ``[min_frequency, max_frequency]``.

        Applied by the optimizer after each update; the sign of each weight is
        kept. ``None`` leaves that side unbounded.
        """

        def __init__(self, min_frequency = None, max_frequency = None):
            self.min_frequency = None if min_frequency is None else float(min_frequency)
            self.max_frequency = None if max_frequency is None else float(max_frequency)

        def __call__(self, w):
            magnitude = ops.abs(w)

            if(self.min_frequency is not None):
                magnitude = ops.maximum(magnitude, self.min_frequency)

            if(self.max_frequency is not None):
                magnitude = ops.minimum(magnitude, self.max_frequency)

            # a weight exactly 0 has sign 0: it is sent to +min_frequency
            sign = ops.where(w >= 0, 1.0, -1.0)
            return sign * magnitude

        def get_config(self):
            return {'min_frequency': self.min_frequency, 'max_frequency': self.max_frequency}


    @keras.saving.register_keras_serializable(package = KERAS_PACKAGE)
    class GatedFAN(keras.layers.Layer):
        """
        Gated Fourier Analysis Network layer.

        Output (``units`` values)::

            [ g * cos(x Wp) , g * sin(x Wp) , (1 - mean(g)) * act(x Wp_bar + b) ]

        * ``Wp`` (``n_inputs x d_p``): learned frequencies. Each periodic unit
          is a sinusoid along a learned direction of the input, with period
          ``2 * pi / |Wp[:, j]|`` in input units.
        * ``g = sigmoid(gate)``: one trainable gate per frequency (starts at
          0.5). It scales the cosine and sine of that frequency; the
          non-periodic part is scaled by ``1 - mean(g)``, so the layer can
          move its capacity between periodic and non-periodic modelling.
        * ``Wp_bar``, ``b``: an ordinary dense layer with activation ``act``.

        With ``gated = False`` the gate is not created and ``g = 1``, ``1 -
        mean(g)`` is replaced by 1 (plain FAN layer). With ``periodic_share =
        0`` the layer is a plain dense layer.

        Parameters
        ----------
        units : int
            Output width.
        periodic_share : float, default 1/3
            Share of the output given to the cosine + sine pairs (see
            :func:`periodic_split`).
        activation : str, default 'gelu'
            Keras activation of the non-periodic part.
        gated : bool, default True
            Create the trainable gates.
        frequency_init_std : float, default 1.0
            Standard deviation of the normal initialisation of ``Wp``. With
            standardized inputs, 1 gives periods of a few standard deviations
            (random Fourier features); use larger values for faster cycles.
        min_frequency, max_frequency : float, optional
            Bounds of ``|Wp|`` (enforced with :class:`FrequencyClip`), e.g.
            to forbid periods longer than the training window.
        """

        def __init__(self, units, periodic_share = 1 / 3, activation = 'gelu', gated = True,
                     frequency_init_std = 1.0, min_frequency = None, max_frequency = None, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)
            self.periodic_share = float(periodic_share)
            self.activation_name = activation
            self.activation = keras.activations.get(activation)
            self.gated = bool(gated)
            self.frequency_init_std = float(frequency_init_std)
            self.min_frequency = min_frequency
            self.max_frequency = max_frequency
            self.d_p, self.d_p_bar = periodic_split(self.units, self.periodic_share)

        def build(self, input_shape):
            n_inputs = int(input_shape[-1])

            if(self.d_p > 0):
                constraint = None
                if((self.min_frequency is not None) or (self.max_frequency is not None)):
                    constraint = FrequencyClip(self.min_frequency, self.max_frequency)

                self.Wp = self.add_weight(name = 'Wp', shape = (n_inputs, self.d_p),
                                          initializer = keras.initializers.RandomNormal(0.0, self.frequency_init_std),
                                          constraint = constraint, trainable = True)

                if(self.gated):
                    # sigmoid(0) = 0.5: periodic and non-periodic parts start with the same weight
                    self.gate = self.add_weight(name = 'gate', shape = (self.d_p,), initializer = 'zeros', trainable = True)

            self.Wp_bar = self.add_weight(name = 'Wp_bar', shape = (n_inputs, self.d_p_bar), initializer = 'glorot_uniform', trainable = True)
            self.b = self.add_weight(name = 'b', shape = (self.d_p_bar,), initializer = 'zeros', trainable = True)

        def call(self, x):
            non_periodic = self.activation(ops.matmul(x, self.Wp_bar) + self.b)

            if(self.d_p == 0):
                return non_periodic

            wx = ops.matmul(x, self.Wp)

            if(self.gated):
                g = ops.sigmoid(self.gate)
                periodic = ops.concatenate([g * ops.cos(wx), g * ops.sin(wx)], axis = -1)
                non_periodic = (1.0 - ops.mean(g)) * non_periodic
            else:
                periodic = ops.concatenate([ops.cos(wx), ops.sin(wx)], axis = -1)

            return ops.concatenate([periodic, non_periodic], axis = -1)

        def compute_output_shape(self, input_shape):
            return tuple(input_shape[:-1]) + (self.units,)

        def gate_values(self):
            """Gates ``g`` (numpy array of ``d_p`` values; ones when the layer is not gated, empty without periodic part)."""
            if(self.d_p == 0):
                return np.array([])
            if(not self.gated):
                return np.ones(self.d_p)
            return keras.ops.convert_to_numpy(ops.sigmoid(self.gate))

        def periods(self):
            """Period of each sinusoid along its own direction, ``2 * pi / |Wp[:, j]|``, in units of the (scaled) input."""
            if(self.d_p == 0):
                return np.array([])
            w = keras.ops.convert_to_numpy(self.Wp)
            return 2 * np.pi / np.maximum(np.linalg.norm(w, axis = 0), 1e-12)

        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units, 'periodic_share': self.periodic_share, 'activation': self.activation_name,
                           'gated': self.gated, 'frequency_init_std': self.frequency_init_std,
                           'min_frequency': self.min_frequency, 'max_frequency': self.max_frequency})
            return config


    _FAN_OBJECTS.update({'GatedFAN': GatedFAN, 'FrequencyClip': FrequencyClip})
    return _FAN_OBJECTS


def make_auc_callback(keras, predict, y_true, rows_mask = None, name = 'val_monitored_auc'):
    """
    Keras callback that adds the ROC AUC of the validation predictions to the logs of every epoch.

    The value is written into the epoch logs under ``name`` BEFORE early
    stopping and checkpoint read them (the callback must come first in the
    callbacks list), so it can be used as ``early_stop_monitor_metric`` /
    ``checkpoint_monitor_metric`` (mode ``'max'``) and appears in the
    training history. With several outputs the AUC is computed on all the
    outputs pooled together. Same function as in fastLSTM.

    Parameters
    ----------
    keras : module
        The Keras module (see :func:`load_keras`).
    predict : callable
        Function without arguments returning the validation predictions,
        shape ``(n_samples, n_outputs)``.
    y_true : array-like
        Actual 0/1 targets aligned with the predictions.
    rows_mask : array-like of bool, optional
        Samples on which the AUC is computed (e.g. the hard cases); all
        samples when ``None``.
    name : str, default 'val_monitored_auc'
        Key of the value in the logs and in the training history.

    Returns
    -------
    keras.callbacks.Callback
        The callback; NaN is logged when the selected samples contain one
        class only.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype = float)
    y_true = y_true.reshape(len(y_true), -1)
    mask = np.ones(len(y_true), dtype = bool) if rows_mask is None else np.asarray(rows_mask, dtype = bool)
    if(len(mask) != len(y_true)):
        raise ValueError(f'rows_mask has {len(mask)} values but there are {len(y_true)} validation samples.')

    class MonitoredAUC(keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs = None):
            predictions = np.asarray(predict(), dtype = float).reshape(len(y_true), -1)
            selected_true, selected_pred = y_true[mask].ravel(), predictions[mask].ravel()
            auc = roc_auc_score(selected_true, selected_pred) if len(np.unique(selected_true)) == 2 else float('nan')
            if(logs is not None):
                logs[name] = auc
            print(f' - {name}: {auc:.4f} ({int(mask.sum())} samples)')

    return MonitoredAUC()


class fastANN:
    """
    Feed-forward (dense) neural network for tabular classification, regression
    and autoencoding.

    The class takes care of the whole life cycle of the model: train/test
    split, feature (and optionally target) scaling, network construction,
    training with early stopping and checkpointing, saving of every artefact
    needed to reproduce the run (model, scalers, training history,
    hyperparameters, data) and reloading them later.

    Its API (hyperparameter names, method names and saved-file layout) is
    shared with the sister package ``fastLSTM``; only ``split_type``,
    ``autoencoder_mode``, the pre-split inputs and the ``'PReLU'`` activation
    are specific to fastANN.

    Network architecture
    --------------------
    ``model_relative_width`` lists the **hidden** Dense layers only. The input
    and the output layers are always added automatically::

        input (n_features)
        Dense(n_features * model_relative_width[0]) + Dropout(model_dropout[0])
        ...
        Dense(n_features * model_relative_width[-1]) + Dropout(model_dropout[-1])
        Dense(output)

    so ``model_relative_width = [2, 1]`` builds two hidden layers with
    ``2 * n_features`` and ``1 * n_features`` units. The output layer has one
    unit per target column (per feature in ``autoencoder_mode``) and
    ``last_layer_activation``.

    Parameters
    ----------
    X_data : pandas.DataFrame, optional
        Features, one row per sample. Required to train a new model unless
        the pre-split arrays are given; can be omitted when the model is
        restored with :meth:`load_all`.
    Y_data : pandas.DataFrame, optional
        Targets aligned with ``X_data``. For binary classification use one
        0/1 column per target.
    X_train_s, X_test_s : array-like, optional
        Already split AND scaled features. Used only when ``X_data`` /
        ``Y_data`` are not given (otherwise they are recomputed by
        :meth:`split_and_scale`).
    Y_train, Y_test : pandas.DataFrame, optional
        Already split targets, used together with ``X_train_s`` /
        ``X_test_s``.
    split_type : {'sequential', 'random'}, default 'sequential'
        * ``'sequential'``: the first ``train_size_rate`` fraction of the rows
          is the training set (use it for time series).
        * ``'random'``: shuffled split with a fixed seed (42).
    model_relative_width : list of float, default [1]
        Width of each hidden layer, relative to the number of features
        (units = ``int(n_features * width)``). Its length is the number of
        hidden layers.
    model_dropout : list of float, default [0]
        Dropout rate (0 - 1) applied after each hidden layer. Must have the
        same length as ``model_relative_width``.
    learning_rate : float, default 0.0003
        Learning rate of the Adam optimizer.
    activation : str, default 'relu'
        Activation of the hidden layers: any Keras activation name
        (``'relu'``, ``'tanh'``, ...) or ``'PReLU'`` (a trainable PReLU layer
        is added after each linear Dense layer). With ``hidden_layer_type =
        'gated_fan'`` it is the activation of the non-periodic part (use
        e.g. ``'gelu'``; ``'PReLU'`` is not available there).
    hidden_layer_type : {'dense', 'gated_fan'}, default 'dense'
        ``'gated_fan'`` replaces every hidden ``Dense`` layer with a gated
        Fourier Analysis Network layer of the same width: ``periodic_share``
        of its outputs are cosines and sines of learned frequencies, scaled
        by trainable gates, the rest is a normal dense part
        (``[g cos(xWp), g sin(xWp), (1 - mean(g)) act(xWp_bar + b)]``). Same
        layer as the sister package ``fastGatedFourierAnalysisNetwork``
        (which adds frequency bounds and initialisations, and the periodic
        components report).
    periodic_share : float, default 1/3
        Share of each gated FAN layer output given to the cosine + sine
        pairs (``hidden_layer_type = 'gated_fan'`` only).
    gated : bool, default True
        Trainable gate per frequency (``'gated_fan'`` only); ``False`` gives
        the ungated FAN layer.
    frequency_init_std : float, default 1.0
        Standard deviation of the normal initialisation of the frequencies
        (``'gated_fan'`` only; inputs are standardized).
    last_layer_activation : str, default 'sigmoid'
        Activation of the output layer: ``'sigmoid'`` for independent binary
        targets, ``'softmax'`` for one-hot multi-class targets, ``'linear'``
        for regression.
    loss : str or keras.losses.Loss, default 'binary_crossentropy'
        Loss function: any Keras loss name (``'binary_crossentropy'``,
        ``'categorical_crossentropy'``, ``'mse'``, ...) or loss object.
    metrics : list of str, default ['accuracy']
        Metrics computed by Keras during training. The names appear in the
        training history, with a ``val_`` prefix for the test set.
    early_stop_monitor_metric : str, default 'val_accuracy'
        Quantity monitored by early stopping (e.g. ``'val_loss'``).
    checkpoint_monitor_metric : str, default 'val_accuracy'
        Quantity monitored to decide which epoch is the best one and must be
        saved (and reloaded at the end of the training). Used only when
        ``save_best_only = True``.
    checkpoint_mode : {'max', 'min', 'auto'}, default 'max'
        Whether ``checkpoint_monitor_metric`` has to be maximised or minimised.
    early_stop_mode : {'max', 'min', 'auto'}, default 'max'
        Whether ``early_stop_monitor_metric`` has to be maximised or
        minimised.
    history_metrics : list of str, default ['accuracy', 'val_accuracy']
        Columns of the training history plotted by
        :meth:`plot_training_history`.
    save_best_only : bool, default True
        If ``True`` the checkpoint overwrites the model file only when the
        monitored quantity improves; otherwise the model is saved at every
        epoch.
    early_stop_patience : int, default 200
        Number of epochs without improvement after which training stops.
    autoencoder_mode : bool, default False
        If ``True`` the network learns to reproduce its (scaled) input: the
        output layer has ``n_features`` units and the targets are the
        features themselves.
    train_size_rate : float, default 0.7
        Fraction (0 - 1) of the rows used for training.
    save_X_Y_data : bool, default True
        If ``True``, ``X_data`` and ``Y_data`` are saved as CSV files when the
        training starts, so that :meth:`load_all` can rebuild the same split.
    data_storage_path : str, default '\\\\cyPredict\\\\'
        Folder where every file is saved and loaded from. It is concatenated
        to the file names as is, so it must end with a path separator.
    model_name : str, default 'ANN'
        Name used in every saved file name.
    scale_targets : bool, default False
        If ``True`` the targets are scaled too (useful for regression);
        :meth:`model_predict` can then bring predictions back to the
        original scale.
    backend : {'tensorflow', 'torch'}, optional
        Deep learning framework that runs the network (through Keras 3).
        ``None`` (default) keeps the backend already active in the Python
        process, or uses the ``KERAS_BACKEND`` environment variable
        (``'tensorflow'`` when it is not set). The backend is fixed for the
        whole process by the first instance: to change it restart the kernel.
        Saved models can be reloaded with either backend.
    sample_weight : array-like, optional
        One weight per row of ``X_data`` (with pre-split inputs: per row of
        ``X_train_s``); the training rows weight the loss, e.g. larger weights
        for the hard cases (options with the strike close to the
        underlying). Validation is not weighted. Not stored in the saved
        files (only whether it was used).
    monitor_auc : bool, default False
        Compute the ROC AUC of the test-set predictions at the end of every
        epoch (binary targets) and log it as ``'val_monitored_auc'``: use it
        as ``early_stop_monitor_metric`` / ``checkpoint_monitor_metric`` with
        mode ``'max'`` to choose the epoch on the AUC instead of the loss or
        the accuracy. Costs one extra prediction pass on the test set per
        epoch.
    monitor_auc_rows : array-like of bool, optional
        One value per row of ``X_data`` (with pre-split inputs: per row of
        ``X_test_s``): the AUC is computed only on the selected test rows
        (e.g. strike within 2% of the underlying). Implies ``monitor_auc =
        True``. Not stored in the saved files (only whether it was used).
    shuffle : bool, default True
        Permute the training rows among the batches at every epoch (Keras
        ``fit(shuffle = ...)``). It changes only the order in which the rows
        are presented to the network, never the train/test split (see
        ``split_type``). ``True`` is the Keras default and the behaviour of
        the previous versions; same parameter name as in ``fastLSTM``.

    Attributes
    ----------
    backend : str
        Active Keras backend (``'tensorflow'`` or ``'torch'``).
    keras : module
        The Keras module used by the instance.
    model : keras.Sequential
        The network (rebuilt by :meth:`network_structure_set_compile`,
        replaced by the saved checkpoint after :meth:`network_training`).
    scaler : sklearn.preprocessing.StandardScaler
        Features scaler.
    Y_scaler : sklearn.preprocessing.StandardScaler or None
        Targets scaler (``None`` unless ``scale_targets = True``).
    X_train, Y_train, X_test, Y_test : pandas.DataFrame
        Unscaled split of the data.
    X_train_s, X_test_s : numpy.ndarray
        Scaled features.
    Y_train_s, Y_test_s : numpy.ndarray or pandas.DataFrame
        Targets used for training (scaled only when ``scale_targets = True``).
    loss_df : pandas.DataFrame
        Training history (one row per epoch).
    hyperparameters : dict
        Everything needed to rebuild the model; saved as JSON at training
        time.

    Examples
    --------
    Binary classification:

    >>> ann = fastANN(X_data = X_df, Y_data = Y_df[['up']],
    ...               model_relative_width = [2, 1],
    ...               model_dropout = [0.2, 0.1],
    ...               data_storage_path = './models/',
    ...               model_name = 'direction')
    >>> ann.network_structure_set_compile()
    >>> ann.network_training(epochs = 200, batch_size = 64)
    >>> results_df, probabilities_df = ann.network_predictions_evaluation(0.5)

    Autoencoder:

    >>> ae = fastANN(X_data = X_df, Y_data = X_df, autoencoder_mode = True,
    ...              model_relative_width = [0.5, 0.25, 0.5],
    ...              model_dropout = [0, 0, 0],
    ...              last_layer_activation = 'linear', loss = 'mse',
    ...              metrics = ['mae'], history_metrics = ['loss', 'val_loss'],
    ...              early_stop_monitor_metric = 'val_loss', early_stop_mode = 'min',
    ...              checkpoint_monitor_metric = 'val_loss', checkpoint_mode = 'min')

    Restoring a trained model:

    >>> ann = fastANN()
    >>> ann.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF ANN MODEL - direction.json',
    ...              file_path_name = './models/')
    """

    def __init__(self,
                 X_data = None,
                 Y_data = None,
                 X_train_s = None,
                 Y_train = None,
                 X_test_s = None,
                 Y_test = None,
                 split_type = 'sequential', # 'random' 'sequential'
                 model_relative_width = [1],
                 model_dropout = [0],
                 learning_rate = 0.0003,
                 activation = 'relu',
                 hidden_layer_type = 'dense',
                 periodic_share = 1 / 3,
                 gated = True,
                 frequency_init_std = 1.0,
                 last_layer_activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'],
                 early_stop_monitor_metric = 'val_accuracy',
                 checkpoint_monitor_metric = 'val_accuracy',
                 checkpoint_mode = 'max',
                 early_stop_mode = 'max',
                 history_metrics = ['accuracy', 'val_accuracy'],
                 save_best_only = True,
                 early_stop_patience = 200,
                 autoencoder_mode = False,
                 train_size_rate = 0.7,
                 save_X_Y_data = True,
                 data_storage_path="\\cyPredict\\",
                 model_name = 'ANN',
                 scale_targets=False,
                 shuffle = True,
                 sample_weight = None,
                 monitor_auc = False,
                 monitor_auc_rows = None,
                 backend = None):

        # Keras with the requested backend (fixed for the whole Python process by the first instance)
        self.keras = load_keras(backend)
        self.backend = self.keras.backend.backend()
        print(f'Keras backend: {self.backend}')

        self.model = self.keras.Sequential()
        self.save_best_only = save_best_only

        self.data_storage_path = data_storage_path
        self.model_name = model_name

        self.model_relative_width = model_relative_width
        self.model_dropout = model_dropout
        self.learning_rate = learning_rate
        self.activation = activation
        if(hidden_layer_type not in ('dense', 'gated_fan')):
            raise ValueError(f"hidden_layer_type must be 'dense' or 'gated_fan', not '{hidden_layer_type}'.")
        self.hidden_layer_type = hidden_layer_type
        self.periodic_share = periodic_share
        self.gated = gated
        self.frequency_init_std = frequency_init_std
        self.last_layer_activation = last_layer_activation
        self.loss = loss
        self.metrics = metrics
        self.early_stop_monitor_metric = early_stop_monitor_metric
        self.checkpoint_monitor_metric = checkpoint_monitor_metric
        self.checkpoint_mode = checkpoint_mode
        self.early_stop_mode = early_stop_mode
        self.early_stop_patience = early_stop_patience
        self.train_size_rate = train_size_rate
        self.autoencoder_mode = autoencoder_mode
        self.history_metrics = history_metrics

        self.hyperparameters_file_name = None

        current_datetime = datetime.now()
        self.model_training_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")

        self.save_X_Y_data = save_X_Y_data

        # names of the files of the last training (set by network_training / load_all)
        self.model_file_name = None
        self.scaler_file_name = None
        self.Y_scaler_file_name = None
        self.training_history_file_name = None
        self.X_data_df_file_name = None
        self.Y_data_df_file_name = None

        self.batch_size = 0

        self.loss_df = pd.DataFrame()

        self.model_summary = {}

        self.X_data = X_data
        self.Y_data = Y_data

        if(X_train_s is None):
            self.X_train_s = pd.DataFrame()
        else:
            self.X_train_s = X_train_s

        if(Y_train is None):
            self.Y_train = pd.DataFrame()
        else:
            self.Y_train = Y_train

        if(X_test_s is None):
            self.X_test_s = pd.DataFrame()
        else:
            self.X_test_s = X_test_s

        if(Y_test is None):
            self.Y_test = pd.DataFrame()
        else:
            self.Y_test = Y_test

        # with pre-split inputs the targets are used as they are
        self.Y_train_s = self.Y_train
        self.Y_test_s = self.Y_test

        self.split_type = split_type
        self.shuffle = shuffle

        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype = float)
        self.monitor_auc_rows = None if monitor_auc_rows is None else np.asarray(monitor_auc_rows, dtype = bool)
        # a rows mask implies the AUC monitor
        self.monitor_auc = bool(monitor_auc) or (self.monitor_auc_rows is not None)
        # with pre-split inputs the arrays already refer to the training / test rows
        self.sample_weight_train = self.sample_weight
        self.monitor_auc_rows_test = self.monitor_auc_rows

        self.scaler = StandardScaler()

        self.scale_targets = scale_targets
        self.Y_scaler = StandardScaler() if scale_targets else None

        self.early_stop_patience_set(self.early_stop_patience)

        if((self.X_data is not None) and (self.Y_data is not None)):
            self.split_and_scale(scaler_fit = True)

        self.hyperparameters = {}

        self.init_hyperparameters()


    def init_hyperparameters(self,
                             model_training_datetime = None,
                             model_file_name = None,
                             scaler_file_name = None,
                             training_history_file_name = None,
                             X_data_df_file_name = None,
                             Y_data_df_file_name = None,
                             Y_scaler_file_name = None
                            ):
        """
        (Re)build the ``hyperparameters`` dictionary from the current attributes.

        The dictionary is what :meth:`save_hyperparameters` writes to JSON and
        what :meth:`set_hyperparameters` reads back, so it contains both the
        network/training settings and the names of the files produced by the
        training. Keys are shared with fastLSTM, plus the ANN-specific
        ``split_type`` and ``autoencoder_mode``.

        Parameters
        ----------
        model_training_datetime : str, optional
            Timestamp (``'%Y-%m-%d %H-%M-%S'``) identifying the training run.
        model_file_name : str, optional
            Name of the ``.keras`` model file.
        scaler_file_name : str, optional
            Name of the ``.pkl`` features scaler file.
        training_history_file_name : str, optional
            Name of the ``.csv`` training history file.
        X_data_df_file_name, Y_data_df_file_name : str, optional
            Names of the ``.csv`` data files (``None`` if not saved).
        Y_scaler_file_name : str, optional
            Name of the ``.pkl`` targets scaler file (``None`` when targets are
            not scaled).

        Returns
        -------
        None
            The result is stored in ``self.hyperparameters``.
        """
        # feature names come from the data when available, otherwise from a loaded configuration
        if(self.X_data is not None):
            X_feature_names = self.X_data.columns.tolist()
        else:
            X_feature_names = getattr(self, 'X_feature_names', None)

        if(self.Y_data is not None):
            Y_feature_names = self.Y_data.columns.tolist()
        else:
            Y_feature_names = getattr(self, 'Y_feature_names', None)

        self.hyperparameters = {
                               'model_training_datetime': model_training_datetime,
                               'model_name': self.model_name,
                               'save_best_only': self.save_best_only,
                               'model_relative_width': self.model_relative_width,
                               'model_dropout': self.model_dropout,
                               'learning_rate': self.learning_rate,
                               'activation': self.activation,
                               'hidden_layer_type': getattr(self, 'hidden_layer_type', 'dense'),
                               'periodic_share': getattr(self, 'periodic_share', 1 / 3),
                               'gated': getattr(self, 'gated', True),
                               'frequency_init_std': getattr(self, 'frequency_init_std', 1.0),
                               'last_layer_activation': self.last_layer_activation,
                               'loss': self.loss if isinstance(self.loss, str) else getattr(self.loss, 'name', str(self.loss)),
                               'metrics': self.metrics,
                               'early_stop_monitor_metric': self.early_stop_monitor_metric,
                               'checkpoint_monitor_metric': self.checkpoint_monitor_metric,
                               'checkpoint_mode': self.checkpoint_mode,
                               'early_stop_mode': self.early_stop_mode,
                               'history_metrics': self.history_metrics,
                               'early_stop_patience': self.early_stop_patience,
                               'autoencoder_mode': self.autoencoder_mode,
                               'train_size_rate': self.train_size_rate,
                               'batch_size': self.batch_size,
                               'X_feature_names': X_feature_names,
                               'Y_feature_names': Y_feature_names,
                               'scaler_type': 'StandardScaler',
                               'split_type': self.split_type,
                               'shuffle': self.shuffle,
                               # arrays not saved: only whether they were used
                               'sample_weight': getattr(self, 'sample_weight', None) is not None,
                               'monitor_auc': getattr(self, 'monitor_auc', False),
                               'monitor_auc_rows': getattr(self, 'monitor_auc_rows', None) is not None,
                               'data_storage_path': self.data_storage_path,
                               'model_file_name': model_file_name,
                               'scaler_file_name': scaler_file_name,
                               'Y_scaler_file_name': Y_scaler_file_name,
                               'training_history_file_name': training_history_file_name,
                               'X_data_df_file_name': X_data_df_file_name,
                               'Y_data_df_file_name': Y_data_df_file_name,
                               'scale_targets': self.scale_targets,
                               # informative: saved models can be reloaded with either backend
                               'backend': self.backend
                              }


    def set_hyperparameters(self):
        """
        Copy the values of ``self.hyperparameters`` back to the attributes.

        Called by :meth:`load_all` after :meth:`load_hyperparameters`. Keys
        missing from the file (older versions) leave the current attribute
        unchanged. The early stopping callback is rebuilt so that it reflects
        the loaded settings.

        ``data_storage_path`` is NOT taken from the file: the folder the
        hyperparameters were loaded from is kept, so a model folder can be
        moved or copied to another machine (the saved value stays in
        ``self.hyperparameters`` for reference).

        Returns
        -------
        None
        """
        self.model_training_datetime = self.hyperparameters['model_training_datetime']
        self.model_name = self.hyperparameters['model_name']
        self.save_best_only = self.hyperparameters['save_best_only']
        self.model_relative_width = self.hyperparameters['model_relative_width']
        self.model_dropout = self.hyperparameters['model_dropout']

        self.learning_rate = self.hyperparameters['learning_rate']
        self.activation = self.hyperparameters['activation']
        # files of versions < 2.3 have dense hidden layers
        self.hidden_layer_type = self.hyperparameters.get('hidden_layer_type', 'dense')
        self.periodic_share = self.hyperparameters.get('periodic_share', 1 / 3)
        self.gated = self.hyperparameters.get('gated', True)
        self.frequency_init_std = self.hyperparameters.get('frequency_init_std', 1.0)
        self.last_layer_activation = self.hyperparameters['last_layer_activation']
        self.loss = self.hyperparameters['loss']

        # keys added in later versions: older files keep the current values
        if('metrics' in self.hyperparameters):
            self.metrics = self.hyperparameters['metrics']

        if('early_stop_monitor_metric' in self.hyperparameters):
            self.early_stop_monitor_metric = self.hyperparameters['early_stop_monitor_metric']

        if('checkpoint_monitor_metric' in self.hyperparameters):
            self.checkpoint_monitor_metric = self.hyperparameters['checkpoint_monitor_metric']

        if('checkpoint_mode' in self.hyperparameters):
            self.checkpoint_mode = self.hyperparameters['checkpoint_mode']

        if('early_stop_mode' in self.hyperparameters):
            self.early_stop_mode = self.hyperparameters['early_stop_mode']

        if('history_metrics' in self.hyperparameters):
            self.history_metrics = self.hyperparameters['history_metrics']

        if('autoencoder_mode' in self.hyperparameters):
            self.autoencoder_mode = self.hyperparameters['autoencoder_mode']

        self.early_stop_patience = self.hyperparameters['early_stop_patience']
        self.train_size_rate = self.hyperparameters['train_size_rate']
        self.batch_size = self.hyperparameters['batch_size']
        self.X_feature_names = self.hyperparameters['X_feature_names']
        self.Y_feature_names = self.hyperparameters['Y_feature_names']

        self.scaler_type = self.hyperparameters['scaler_type']
        self.split_type = self.hyperparameters['split_type']
        # files of versions < 2.1 do not store shuffle: Keras shuffled by default
        self.shuffle = self.hyperparameters.get('shuffle', True)
        # data_storage_path is deliberately not restored (see docstring): the saved one may not exist any more
        self.model_file_name = self.hyperparameters['model_file_name']
        self.scaler_file_name = self.hyperparameters['scaler_file_name']
        self.Y_scaler_file_name = self.hyperparameters.get('Y_scaler_file_name')

        self.training_history_file_name = self.hyperparameters['training_history_file_name']
        self.X_data_df_file_name = self.hyperparameters['X_data_df_file_name']
        self.Y_data_df_file_name = self.hyperparameters['Y_data_df_file_name']

        self.scale_targets = self.hyperparameters.get('scale_targets', False)
        if(self.scale_targets and (self.Y_scaler is None)):
            self.Y_scaler = StandardScaler()

        # callbacks depend on the loaded settings
        self.early_stop_patience_set(self.early_stop_patience)


    def compile_metrics(self):
        """
        Metrics passed to ``model.compile``, with ``'accuracy'`` made explicit.

        With more than one output Keras 3 turns the string ``'accuracy'``
        into categorical accuracy (argmax across the outputs), which is wrong
        for independent sigmoid outputs: on multi-step or multi-target binary
        targets a constant model can score above 90%. ``'accuracy'`` (or
        ``'acc'``) is therefore replaced by ``BinaryAccuracy`` when the output
        activation is sigmoid and by ``CategoricalAccuracy`` when it is
        softmax; the metric keeps the name ``'accuracy'``, so the history
        columns (``accuracy``, ``val_accuracy``) do not change.

        Returns
        -------
        list
            Metrics for ``model.compile``.
        """
        resolved = []

        for metric in self.metrics:
            if((metric in ('accuracy', 'acc')) and (self.last_layer_activation == 'sigmoid')):
                resolved.append(self.keras.metrics.BinaryAccuracy(name = 'accuracy'))
            elif((metric in ('accuracy', 'acc')) and (self.last_layer_activation == 'softmax')):
                resolved.append(self.keras.metrics.CategoricalAccuracy(name = 'accuracy'))
            else:
                resolved.append(metric)

        return resolved


    def early_stop_patience_set(self, patience = None):
        """
        Create the early stopping callback (``self.early_stop``).

        It monitors ``early_stop_monitor_metric`` in ``early_stop_mode``.

        Parameters
        ----------
        patience : int, optional
            Number of epochs without improvement before stopping. When given
            it also updates ``self.early_stop_patience``; when ``None`` the
            current value is used.

        Returns
        -------
        None

        Examples
        --------
        >>> ann.early_stop_patience_set(50)
        """
        if(patience is not None):
            self.early_stop_patience = patience

        self.early_stop = self.keras.callbacks.EarlyStopping(monitor = self.early_stop_monitor_metric,
                                        mode = self.early_stop_mode,
                                        verbose = 1,
                                        patience = self.early_stop_patience)


    def checkpoint_callback(self, save_best_only = None):
        """
        Create the model checkpoint callback (``self.model_checkpoint``).

        The model is saved to ``data_storage_path + model_file_name`` (the
        name stored in the hyperparameters by :meth:`network_training`, or a
        timestamped default) monitoring ``checkpoint_monitor_metric`` in
        ``checkpoint_mode``.

        Parameters
        ----------
        save_best_only : bool, optional
            If ``True`` the file is overwritten only when the monitored
            quantity improves. Defaults to ``self.save_best_only``.

        Returns
        -------
        keras.callbacks.ModelCheckpoint
            The callback (also stored in ``self.model_checkpoint``).
        """
        if(save_best_only is None):
            save_best_only = self.save_best_only

        if(self.hyperparameters.get('model_file_name')):
            model_file_name = self.hyperparameters['model_file_name']
        else:
            model_file_name = self.model_training_datetime + ' - ANN MODEL - ' + self.model_name + '.keras'

        # the callback is stored in its own attribute: assigning it to self.checkpoint_callback
        # would shadow this method and make a second training fail
        self.model_checkpoint = self.keras.callbacks.ModelCheckpoint(self.data_storage_path + model_file_name,
                                                monitor = self.checkpoint_monitor_metric,
                                                mode = self.checkpoint_mode,
                                                verbose = 1,
                                                save_best_only = save_best_only)

        return self.model_checkpoint


    def network_structure_set_compile(self):
        """
        Build and compile the network.

        One ``Dense`` (+ ``PReLU`` when ``activation = 'PReLU'``) +
        ``Dropout`` block is added for each element of
        ``model_relative_width``; the output layer is added automatically
        (see the class docstring). The network is compiled with
        Adam(``learning_rate``), ``loss`` and ``metrics``.

        Returns
        -------
        None
            The network is stored in ``self.model`` and its text summary in
            ``self.model_summary``.

        Raises
        ------
        ValueError
            If there is no training data or if ``model_dropout`` and
            ``model_relative_width`` have different lengths.

        Examples
        --------
        >>> ann.model_relative_width = [3, 2, 1]   # three hidden layers
        >>> ann.model_dropout = [0.2, 0.2, 0.1]
        >>> ann.network_structure_set_compile()
        """
        if(len(self.X_train_s) == 0):
            raise ValueError('No training data: pass X_data and Y_data (or the pre-split data) to the constructor or call split_and_scale first.')

        if(len(self.model_dropout) != len(self.model_relative_width)):
            raise ValueError(f'model_dropout ({len(self.model_dropout)} values) and model_relative_width '
                             f'({len(self.model_relative_width)} values) must have the same length.')

        n_features = self.X_train_s.shape[1]

        keras = self.keras

        # reset
        self.model = keras.Sequential()

        # input layer: one vector of n_features values per sample
        self.model.add(keras.Input(shape = (n_features,)))

        # hidden layers
        for i in range(len(self.model_relative_width)):

            model_relative_width = self.model_relative_width[i]
            model_dropout = self.model_dropout[i]

            if(getattr(self, 'hidden_layer_type', 'dense') == 'gated_fan'):
                # periodic (cos, sin) and non-periodic units side by side, see fan_layers
                self.model.add(fan_layers(keras)['GatedFAN'](units = int(n_features * model_relative_width),
                                                             periodic_share = self.periodic_share, activation = self.activation,
                                                             gated = self.gated, frequency_init_std = self.frequency_init_std,
                                                             name = f'gated_fan_{i}'))
            elif(self.activation != 'PReLU'):
                self.model.add(keras.layers.Dense(int(n_features * model_relative_width), activation = self.activation))
            else:
                # PReLU has trainable parameters, so it is a layer and not an activation name
                self.model.add(keras.layers.Dense(int(n_features * model_relative_width)))
                self.model.add(keras.layers.PReLU())

            self.model.add(keras.layers.Dropout(model_dropout))

        if self.autoencoder_mode:
            # last layer: ensure it matches input size in autoencoder mode
            self.model.add(keras.layers.Dense(n_features, activation=self.last_layer_activation))

        else:
            # last layer: ensure it matches targets size in not autoencoder mode
            self.model.add(keras.layers.Dense(int(self.Y_train.shape[1]), activation = self.last_layer_activation))

        # compile
        self.model.compile(optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate),
                         loss = self.loss,
                         metrics = self.compile_metrics())

        # report: model.summary() only prints, so its lines are collected to keep a copy in model_summary
        summary_lines = []
        self.model.summary(print_fn = lambda line, *args, **kwargs: summary_lines.append(line))
        self.model_summary = '\n'.join(summary_lines)

        print(self.model_summary)


    def auc_callbacks(self, X_val, Y_val):
        """
        The test-set AUC monitor (see ``monitor_auc``), as a list ready for ``fit``.

        Parameters
        ----------
        X_val, Y_val : array-like
            Validation inputs and the targets the network is trained on.

        Returns
        -------
        list of keras.callbacks.Callback
            Empty when ``monitor_auc`` is off; otherwise the callback built by
            :func:`make_auc_callback`, restricted to ``monitor_auc_rows``.
        """
        if(not getattr(self, 'monitor_auc', False)):
            return []

        return [make_auc_callback(self.keras, lambda: self.model.predict(X_val, verbose = 0), np.asarray(Y_val),
                                  getattr(self, 'monitor_auc_rows_test', None))]


    def network_training(self, epochs, batch_size, callbacks = None):
        """
        Train the network and save every artefact of the run.

        Steps:

        1. a timestamp identifies the run and all the file names;
        2. ``X_data`` / ``Y_data`` are saved as CSV (if ``save_X_Y_data``);
        3. hyperparameters (JSON) and scalers (``.pkl``) are saved;
        4. the network is trained with early stopping and checkpointing (in
           ``autoencoder_mode`` the targets are the scaled features);
        5. the training history is saved (CSV), the saved checkpoint is
           reloaded into ``self.model`` (the best epoch when
           ``save_best_only = True``, the last one otherwise) and the history
           is plotted.

        Files written in ``data_storage_path`` (``<dt>`` = timestamp,
        ``<name>`` = ``model_name``)::

            <dt> - ANN MODEL - <name>.keras
            <dt> - SCALER FOR ANN MODEL - <name>.pkl
            <dt> - Y SCALER FOR ANN MODEL - <name>.pkl        (scale_targets only)
            <dt> - TRAINING HISTORY OF ANN MODEL - <name>.csv
            <dt> - HYPERPARAMETERS OF ANN MODEL - <name>.json
            <dt> - X_data FOR ANN MODEL - <name>.csv          (save_X_Y_data only)
            <dt> - Y_data FOR ANN MODEL - <name>.csv          (save_X_Y_data only)

        Parameters
        ----------
        epochs : int
            Maximum number of epochs (early stopping may end the training
            before).
        batch_size : int
            Number of samples per gradient update.
        callbacks : list of keras.callbacks.Callback, optional
            Extra callbacks run together with the early stopping and the
            best-epoch checkpoint (e.g. a time limit or a learning-rate
            schedule).

        Returns
        -------
        None
            The trained model is in ``self.model``, the history in
            ``self.loss_df``.

        Raises
        ------
        ValueError
            If the network has not been built (checked before writing any
            file).

        Examples
        --------
        >>> ann.network_structure_set_compile()
        >>> ann.network_training(epochs = 300, batch_size = 64)
        """
        if(len(self.model.layers) == 0):
            raise ValueError('The network is not built: call network_structure_set_compile() before training.')

        self.batch_size = batch_size

        current_datetime = datetime.now()
        self.model_training_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")

        model_file_name = self.model_training_datetime + ' - ANN MODEL - ' + self.model_name + '.keras'
        scaler_file_name = self.model_training_datetime + ' - SCALER FOR ANN MODEL - ' + self.model_name +'.pkl'
        Y_scaler_file_name = self.model_training_datetime + ' - Y SCALER FOR ANN MODEL - ' + self.model_name +'.pkl' if self.scale_targets else None
        training_history_file_name = self.model_training_datetime + ' - TRAINING HISTORY OF ANN MODEL - ' + self.model_name + '.csv'
        hyperparameters_file_name = self.model_training_datetime + ' - HYPERPARAMETERS OF ANN MODEL - ' + self.model_name + ".json"
        self.hyperparameters_file_name = hyperparameters_file_name

        # X and Y data are saved so that load_all can rebuild the same train/test split
        if(self.save_X_Y_data and (self.X_data is not None) and (self.Y_data is not None)):
            X_data_df_file_name = self.model_training_datetime + ' - X_data FOR ANN MODEL - ' + self.model_name + '.csv'
            Y_data_df_file_name = self.model_training_datetime + ' - Y_data FOR ANN MODEL - ' + self.model_name + '.csv'

            # index=False: the index is not saved
            self.X_data.to_csv(self.data_storage_path + X_data_df_file_name, index=False)
            self.Y_data.to_csv(self.data_storage_path + Y_data_df_file_name, index=False)

            print('\nsave_X_Y_data saved.')

        else:
            X_data_df_file_name = None
            Y_data_df_file_name = None

            print('\nsave_X_Y_data not saved.')

        self.model_file_name = model_file_name
        self.scaler_file_name = scaler_file_name
        self.Y_scaler_file_name = Y_scaler_file_name
        self.training_history_file_name = training_history_file_name
        self.X_data_df_file_name = X_data_df_file_name
        self.Y_data_df_file_name = Y_data_df_file_name

        # save hyperparameters
        print('\nInit hyperparameters.')
        self.init_hyperparameters(
                                  model_training_datetime = self.model_training_datetime,
                                  model_file_name = model_file_name,
                                  scaler_file_name = scaler_file_name,
                                  training_history_file_name = training_history_file_name,
                                  X_data_df_file_name = X_data_df_file_name,
                                  Y_data_df_file_name = Y_data_df_file_name,
                                  Y_scaler_file_name = Y_scaler_file_name
                                 )

        print('\nSave hyperparameters.')
        self.save_hyperparameters(hyperparameters_file_name)

        # save used scalers
        joblib.dump(self.scaler, self.data_storage_path + scaler_file_name)

        if(self.scale_targets == True):
            joblib.dump(self.Y_scaler, self.data_storage_path + Y_scaler_file_name)

        self.checkpoint_callback(self.save_best_only)

        if self.autoencoder_mode:
            # autoencoder: the network reproduces its scaled input, so the scaled features are also the targets
            X_train = self.X_train_s
            Y_train = self.X_train_s
            X_val = self.X_test_s
            Y_val = self.X_test_s
        else:
            # normal mode: scaled inputs, targets scaled only when scale_targets is True
            X_train = self.X_train_s
            Y_train = self.Y_train_s if self.scale_targets else self.Y_train
            X_val = self.X_test_s
            Y_val = self.Y_test_s if self.scale_targets else self.Y_test

        # model training
        history = self.model.fit(
            x=X_train,
            y=Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=self.shuffle,
            sample_weight=getattr(self, 'sample_weight_train', None),
            # the AUC monitor comes first: it writes val_monitored_auc into the logs read by early stopping and checkpoint
            callbacks=self.auc_callbacks(X_val, Y_val) + [self.early_stop, self.model_checkpoint] + list(callbacks or [])
        )

        # save history
        self.loss_df = pd.DataFrame(history.history)
        self.loss_df.to_csv(self.data_storage_path + training_history_file_name)

        # keep the model saved by the checkpoint (best epoch when save_best_only)
        self.load_model(model_file_name)

        # plot history
        self.plot_training_history()


    def save_hyperparameters(self, file_name):
        """
        Save ``self.hyperparameters`` as JSON in ``data_storage_path``.

        NumPy values are converted to plain Python values; anything else that
        JSON cannot represent is saved as its string.

        Parameters
        ----------
        file_name : str
            Name of the JSON file (without the folder).

        Returns
        -------
        None
        """
        # serialised before opening the file, so that an error cannot leave a truncated JSON
        text = json.dumps(self.hyperparameters, default = lambda value: value.tolist() if hasattr(value, 'tolist') else str(value))

        with open(self.data_storage_path + file_name, "w") as file:
            file.write(text)


    def load_hyperparameters(self, file_name, file_path_name = None):
        """
        Load a hyperparameters JSON file into ``self.hyperparameters``.

        The attributes are NOT updated: call :meth:`set_hyperparameters`
        afterwards (or use :meth:`load_all`).

        Parameters
        ----------
        file_name : str
            Name of the JSON file.
        file_path_name : str, optional
            Folder of the file; when given it replaces ``data_storage_path``.

        Returns
        -------
        None
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        self.hyperparameters_file_name = file_name

        print(f'\nTrying to load hyperparameters {self.data_storage_path + file_name}')
        with open(self.data_storage_path + file_name, "r") as file:
            self.hyperparameters = json.load(file)
        print(f'Hyperparameters loaded.')


    def load_model(self, model_file_name = None, file_path_name = None):
        """
        Load a saved Keras model into ``self.model``.

        The model can have been trained with either backend. It is loaded
        without its saved compile state and recompiled with the current
        ``loss``, ``metrics`` and ``learning_rate`` (restored from the
        hyperparameters by :meth:`load_all`), so a further training starts
        with a fresh optimizer state.

        Parameters
        ----------
        model_file_name : str, optional
            Name of the ``.keras`` file. Defaults to the one stored in the
            hyperparameters.
        file_path_name : str, optional
            Folder of the file; when given it replaces ``data_storage_path``.

        Returns
        -------
        keras.Model
            The loaded model (also stored in ``self.model``).
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(model_file_name is None):
            model_file_name = self.hyperparameters['model_file_name']

        model_file_path = self.data_storage_path + model_file_name
        if os.path.exists(model_file_path):
            print("Model file exists.")
            file_size = os.path.getsize(model_file_path)
            print(f"File size: {file_size} bytes")
            if file_size == 0:
                print("Warning: The model file is empty.")
        else:
            print("Error: Model file does not exist.")

        print(f'\nTrying to load model {model_file_path}')
        # compile = False: the saved compile configuration can refer to backend-specific classes (e.g. the PyTorch
        # Adam optimizer) that cannot be loaded with the other backend. Architecture and weights are portable, so the
        # model is loaded without it and recompiled with the current loss, metrics and learning rate.
        # custom objects: models with gated FAN hidden layers
        self.model = self.keras.models.load_model(model_file_path, compile = False, custom_objects = fan_layers(self.keras))
        self.model.compile(optimizer = self.keras.optimizers.Adam(learning_rate = self.learning_rate),
                           loss = self.loss,
                           metrics = self.compile_metrics())
        print(f'Model loaded.')

        if self.model:
            print("Model is correctly loaded and accessible.")
            self.model.summary()
        else:
            print("Model is None after loading. Check the loading logic.")

        return self.model


    def model_predict(self, data, apply_scaler=True, descale_result=True):
        """
        Predict with the trained network.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            Samples of shape ``(n_samples, n_features)``, same columns as
            ``X_data``.
        apply_scaler : bool, default True
            If ``True`` the features are scaled with ``self.scaler`` (pass
            ``False`` for data already scaled).
        descale_result : bool, default True
            If ``True`` and ``scale_targets = True``, predictions are brought
            back to the original scale with ``self.Y_scaler``.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(n_samples, n_outputs)``.

        Examples
        --------
        >>> ann.model_predict(X_df.tail(5))
        """
        if apply_scaler:
            print('Scaler applied.')
            data = self.scaler.transform(data)

        predictions = self.model.predict(data)

        if descale_result and self.scale_targets and (self.Y_scaler is not None):
            predictions = self.Y_scaler.inverse_transform(predictions)

        return predictions


    def load_scaler(self, scaler_file_name = None, Y_scaler_file_name = None, file_path_name = None):
        """
        Load the features scaler and, if targets are scaled, the targets scaler.

        Parameters
        ----------
        scaler_file_name : str, optional
            Features scaler file. Defaults to the name stored in the
            hyperparameters, or to the standard timestamped name.
        Y_scaler_file_name : str, optional
            Targets scaler file (used only when ``scale_targets = True``).
            Same defaults as ``scaler_file_name``.
        file_path_name : str, optional
            Folder of the files; when given it replaces ``data_storage_path``.

        Returns
        -------
        None
            Scalers are stored in ``self.scaler`` and ``self.Y_scaler``.
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(scaler_file_name is None):
            scaler_file_name = (self.hyperparameters.get('scaler_file_name')
                                or self.model_training_datetime + ' - SCALER FOR ANN MODEL - ' + self.model_name +'.pkl')

        if(Y_scaler_file_name is None):
            Y_scaler_file_name = (self.hyperparameters.get('Y_scaler_file_name')
                                  or self.model_training_datetime + ' - Y SCALER FOR ANN MODEL - ' + self.model_name +'.pkl')

        print(f"\nTrying to load scaler {self.data_storage_path}{scaler_file_name}")
        self.scaler = joblib.load(self.data_storage_path + scaler_file_name)
        print(f'Scaler loaded.')

        if(self.scale_targets == True):
            print(f"\nTrying to load Y_scaler {self.data_storage_path}{Y_scaler_file_name}")
            self.Y_scaler = joblib.load(self.data_storage_path + Y_scaler_file_name)
            print(f'Y_scaler loaded.')


    def load_training_history(self, training_history_file_name = None, file_path_name = None):
        """
        Load a training history CSV into ``self.loss_df``.

        Parameters
        ----------
        training_history_file_name : str, optional
            Name of the CSV file. Defaults to the one stored in the
            hyperparameters.
        file_path_name : str, optional
            Folder of the file; when given it replaces ``data_storage_path``.

        Returns
        -------
        None
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(training_history_file_name is None):
            training_history_file_name = self.hyperparameters['training_history_file_name']

        print(f"\nTrying to load training history {self.data_storage_path + training_history_file_name}")
        # the first column is the epoch index written by DataFrame.to_csv
        self.loss_df = pd.read_csv(self.data_storage_path + training_history_file_name, index_col = 0)
        print(f'Training history loaded.')


    def load_all(self, hyperparameters_file_name = None, file_path_name = None):
        """
        Restore a trained model with everything that was saved with it.

        Loads, in order: hyperparameters (and applies them), model, scalers,
        training history and, if they were saved, ``X_data`` / ``Y_data``,
        which are split and scaled again with the loaded scaler (no refit),
        so that evaluation methods work immediately.

        Parameters
        ----------
        hyperparameters_file_name : str, optional
            Name of the hyperparameters JSON file. Defaults to
            ``self.hyperparameters_file_name`` (the last training).
        file_path_name : str, optional
            Folder of the files; when given it replaces ``data_storage_path``.

        Returns
        -------
        None

        Examples
        --------
        >>> ann = fastANN()
        >>> ann.load_all('2025-03-07 10-00-00 - HYPERPARAMETERS OF ANN MODEL - ANN.json',
        ...              file_path_name = './models/')
        >>> ann.network_predictions_evaluation(0.5)
        """
        if(file_path_name is not None):
            self.data_storage_path = file_path_name

        if(hyperparameters_file_name is not None):
            self.hyperparameters_file_name = hyperparameters_file_name

        print(f'\nTrying to open hyperparameters file {self.hyperparameters_file_name}')
        self.load_hyperparameters(self.hyperparameters_file_name)

        print(f'\nSetting hyperparameters')
        self.set_hyperparameters()

        print(f'Hyperparameters:\n')
        display(self.hyperparameters)

        self.load_model(self.hyperparameters['model_file_name'])

        print(f"\nTrying to import ANN scalers")
        self.load_scaler()

        print(f"\nTrying to import ANN training history {self.hyperparameters['training_history_file_name']}")
        self.load_training_history(self.hyperparameters['training_history_file_name'])

        # load X and Y data, split and scale
        if((self.hyperparameters['X_data_df_file_name'] is not None) and (self.hyperparameters['Y_data_df_file_name'] is not None)):

            self.X_data = pd.read_csv(self.data_storage_path + self.hyperparameters['X_data_df_file_name'])
            self.Y_data = pd.read_csv(self.data_storage_path + self.hyperparameters['Y_data_df_file_name'])

            self.split_and_scale(scaler_fit = False) # scaler is loaded, no need to fit it again

        if self.model:
            print("Model is STILL correctly loaded and accessible.")
            self.model.summary()
        else:
            print("Model is None after loading. Check the loading logic.")


    def network_predictions_evaluation(self, min_probability, output_dict = False):
        """
        Evaluate a binary classifier on the test set.

        Predicted probabilities are turned into 0/1 with the threshold
        ``min_probability`` and compared with the actual targets through
        sklearn's ``classification_report`` (printed for every target
        column).

        Parameters
        ----------
        min_probability : float
            Threshold (0 - 1): predictions strictly greater than it become 1.
        output_dict : bool, default False
            If ``True`` the report is also returned as a dictionary.

        Returns
        -------
        filtered_predictions_results_df : pandas.DataFrame
            0/1 predictions, one column per target.
        predictions_df : pandas.DataFrame
            Raw predicted probabilities (columns numbered from 0).
        report : dict
            Only when ``output_dict = True``: classification report of the
            LAST target column.

        Examples
        --------
        >>> results_df, probabilities_df = ann.network_predictions_evaluation(0.6)
        >>> results_df, probabilities_df, report = ann.network_predictions_evaluation(0.6, output_dict = True)
        >>> report['1']['precision']
        """
        print(f'len X_test_s {len(self.X_test_s)}')
        print(f'len Y_test {len(self.Y_test)}')

        # Cut off predictions with low probability
        predictions = self.model.predict(self.X_test_s)
        predictions_df = pd.DataFrame(predictions)
        filtered_predictions_results_df = pd.DataFrame()

        report = None
        for count, col_name in enumerate(self.Y_test.columns):

            filtered_predictions_results_df[col_name] = predictions_df[count].apply(lambda x: 1 if x > min_probability else 0 ).values
            report = classification_report(self.Y_test[col_name], filtered_predictions_results_df[col_name], output_dict = output_dict)

            print(report)

        if(output_dict == False):
            return filtered_predictions_results_df, predictions_df

        elif(output_dict == True):
            return filtered_predictions_results_df, predictions_df, report


    def plot_training_history(self):
        """
        Plot the ``history_metrics`` columns of the training history (``self.loss_df``).

        Returns
        -------
        None
        """
        self.loss_df[self.history_metrics].plot()


    def split_and_scale(self, scaler_fit = False):
        """
        Split the data into training and test set and scale them.

        The split follows ``split_type`` (``'sequential'``: first
        ``train_size_rate`` fraction of the rows for training; ``'random'``:
        shuffled split with seed 42 and the same proportion). Features are
        scaled with ``self.scaler``; targets with ``self.Y_scaler`` only when
        ``scale_targets = True``.

        Parameters
        ----------
        scaler_fit : bool, default False
            If ``True`` the scalers are fitted on the training set (new
            data); if ``False`` the already fitted (e.g. loaded) scalers are
            only applied.

        Returns
        -------
        None
            Results are stored in ``X_train``, ``Y_train``, ``X_test``,
            ``Y_test`` (DataFrames), ``X_train_s``, ``X_test_s`` (numpy
            arrays) and ``Y_train_s``, ``Y_test_s``.

        Examples
        --------
        >>> ann.X_data, ann.Y_data = new_X_df, new_Y_df
        >>> ann.split_and_scale(scaler_fit = True)
        """
        print(f'self.split_type {self.split_type}')

        if(self.split_type == 'sequential'):
            train_size = int(len(self.X_data) * self.train_size_rate)
            test_size = len(self.X_data) - train_size
            train_positions, test_positions = np.arange(train_size), np.arange(train_size, len(self.X_data))

            self.X_train = self.X_data.head(train_size)
            self.Y_train = self.Y_data.head(train_size)
            self.X_test = self.X_data.tail(test_size)
            self.Y_test = self.Y_data.tail(test_size)

            print('split_and_scale, SEQUENTIAL split')

        if(self.split_type == 'random'):
            # train_size_rate is the TRAINING fraction, as for the sequential split
            # positions are split with the same seed, so the rows are the ones of the previous versions
            train_positions, test_positions = train_test_split(np.arange(len(self.X_data)), train_size = self.train_size_rate,
                                                               random_state = 42)
            self.X_train, self.X_test = self.X_data.iloc[train_positions], self.X_data.iloc[test_positions]
            self.Y_train, self.Y_test = self.Y_data.iloc[train_positions], self.Y_data.iloc[test_positions]
            print('split_and_scale, RANDOM split')

        for name, values in [('sample_weight', getattr(self, 'sample_weight', None)), ('monitor_auc_rows', getattr(self, 'monitor_auc_rows', None))]:
            if((values is not None) and (len(values) != len(self.X_data))):
                raise ValueError(f'{name} has {len(values)} values but X_data has {len(self.X_data)} rows.')
        self.sample_weight_train = None if getattr(self, 'sample_weight', None) is None else self.sample_weight[train_positions]
        self.monitor_auc_rows_test = None if getattr(self, 'monitor_auc_rows', None) is None else self.monitor_auc_rows[test_positions]

        # scale
        if(scaler_fit == True):
            print('\tFit transform X_train')
            self.X_train_s = self.scaler.fit_transform(self.X_train)
        else:
            print('\tOnly transform X_train')
            self.X_train_s = self.scaler.transform(self.X_train)

        print('\tTransform X_test')
        self.X_test_s = self.scaler.transform(self.X_test)

        if self.scale_targets:
            # a targets scaler created here (scale_targets switched on later) has never been fitted
            fit_Y_scaler = scaler_fit
            if(self.Y_scaler is None):
                self.Y_scaler = StandardScaler()
                fit_Y_scaler = True

            if fit_Y_scaler:
                self.Y_train_s = self.Y_scaler.fit_transform(self.Y_train)
            else:
                self.Y_train_s = self.Y_scaler.transform(self.Y_train)

            self.Y_test_s = self.Y_scaler.transform(self.Y_test)
        else:
            self.Y_train_s = self.Y_train
            self.Y_test_s = self.Y_test

        print('split_and_scale, end data length')
        print(f"\tShape of X_train: {self.X_train.shape}")
        print(f"\tShape of Y_train: {self.Y_train.shape}")
        print(f"\tShape of X_test: {self.X_test.shape}")
        print(f"\tShape of Y_test: {self.Y_test.shape}")


    def binary_precision_recall_vs_scoring(self, n_points = 15, plot = True):
        """
        Precision and recall of class ``1`` as a function of the probability cutoff.

        :meth:`network_predictions_evaluation` is run for every cutoff from
        ``n_points / 100`` to ``0.99`` with step ``0.01``.

        Parameters
        ----------
        n_points : int, default 15
            Despite the name, it is the FIRST cutoff expressed in percent
            (15 -> cutoffs 0.15, 0.16, ..., 0.99).
        plot : bool, default True
            If ``True`` an interactive Plotly chart is shown.

        Returns
        -------
        pandas.DataFrame
            Columns ``'Cutoff'``, ``'Precision'``, ``'Recall'``.

        Notes
        -----
        The values refer to the last target column, and the class labels of
        ``Y_data`` must be the integers 0/1 (the report keys are ``'0'`` and
        ``'1'``; float labels would produce ``'1.0'``).

        Examples
        --------
        >>> pr_df = ann.binary_precision_recall_vs_scoring(n_points = 30, plot = False)
        >>> pr_df.loc[pr_df['Precision'].idxmax()]
        """
        # lists collecting precision and recall for every cutoff
        precision_list = []
        recall_list = []
        cutoff_values = []

        # evaluate every cutoff from n_points% to 99%
        for cutoff in range(n_points, 100, 1):
            cutoff_value = cutoff / 100  # cutoff as a probability (0 - 1)
            print(f'Evaluating cutoff value = {cutoff_value}')

            filtered_predictions_results_df, predictions_df, dictionary = self.network_predictions_evaluation(cutoff_value, output_dict=True)

            # precision and recall of the positive class
            precision_list.append(dictionary['1']['precision'])
            recall_list.append(dictionary['1']['recall'])
            cutoff_values.append(cutoff_value)

        df = pd.DataFrame({'Cutoff': cutoff_values, 'Precision': precision_list, 'Recall': recall_list})

        if(plot == True):

            # precision and recall vs cutoff (interactive Plotly chart)
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Precision'], mode='lines', name='Precision'))

            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Recall'], mode='lines', name='Recall'))

            fig.update_layout(
                xaxis_title='Cutoff',
                yaxis_title='Value',
                title='Precision and Recall vs Cutoff'
            )

            fig.show()

        return df


    def compute_gradients(self, inputs, targets):
        """
        Gradient of the mean squared error with respect to the network inputs.

        Works with both backends (TensorFlow ``GradientTape`` or PyTorch
        autograd).

        Parameters
        ----------
        inputs : numpy.ndarray
            Input samples, shape ``(n_samples, n_features)``.
        targets : array-like
            Targets, shape ``(n_samples, n_outputs)``.

        Returns
        -------
        numpy.ndarray
            Gradients with the same shape as ``inputs``.
        """
        keras = self.keras

        # MSE is used for every network type: only the gradient magnitude matters here
        mse = keras.losses.MeanSquaredError()
        targets = keras.ops.convert_to_tensor(np.asarray(targets, dtype = np.float32))

        if(self.backend == 'torch'):
            # PyTorch autograd: the inputs become a leaf tensor that records its gradient
            inputs = keras.ops.convert_to_tensor(np.asarray(inputs, dtype = np.float32))
            inputs.requires_grad_(True)
            loss = mse(targets, self.model(inputs))
            loss.backward()
            return inputs.grad.detach().cpu().numpy()

        import tensorflow
        inputs = tensorflow.convert_to_tensor(np.asarray(inputs, dtype = np.float32))
        with tensorflow.GradientTape() as tape:
            tape.watch(inputs)
            loss = mse(targets, self.model(inputs))

        return tape.gradient(loss, inputs).numpy()


    def gradient_feature_importance(self, feature_names = None):
        """
        Gradient-based feature importance computed on the test set.

        The importance of a feature is the mean absolute gradient of the loss
        with respect to that input, averaged over all test samples, then
        normalised to sum to 1. A horizontal Plotly bar chart is shown.

        Parameters
        ----------
        feature_names : list of str, optional
            Names of the features, in the column order of ``X_data``.
            Defaults to the columns of ``X_data`` (or to the saved
            ``X_feature_names``).

        Returns
        -------
        sorted_features_importance : list of float
            Normalised importances in increasing order.
        sorted_features_names : list of str
            Feature names in the same order.

        Examples
        --------
        >>> importance, names = ann.gradient_feature_importance()
        >>> names[-1]   # most important feature
        """
        if(feature_names is None):
            feature_names = self.X_data.columns.tolist() if self.X_data is not None else self.X_feature_names


        # the targets the network was trained on: scaled features (autoencoder) or (possibly scaled) targets
        targets = self.X_test_s if self.autoencoder_mode else self.Y_test_s

        gradients = self.compute_gradients(self.X_test_s, targets)

        # average over samples: one value per feature
        feature_importance = np.mean(np.abs(gradients), axis=0)

        feature_importance = feature_importance / np.sum(feature_importance)

        # sort features by importance (increasing, so that the most important is on top of the bar chart)
        sorted_idx = np.argsort(feature_importance)
        sorted_features_names = [feature_names[i] for i in sorted_idx]
        sorted_features_importance = [feature_importance[i] for i in sorted_idx]

        fig = go.Figure(go.Bar(
            x=sorted_features_importance,
            y=sorted_features_names,
            orientation='h'
        ))
        fig.update_layout(
            title='Feature Importance (Gradient-based)',
            xaxis_title='Normalized Importance',
            yaxis_title='Features',
            height=800
        )
        fig.show()

        return sorted_features_importance, sorted_features_names
