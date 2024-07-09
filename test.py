#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import numpy

import problems
import plots
from learner import PyTorchLearner, GPyTorchLearner, ScikitLearnLearner


# Prepare dataset
noise = 0.5
# (y, x), func = problems.generate_linear_datasets(100, noise=noise)
(y, x), func = problems.generate_sigmoid_datasets(100, noise=noise)
# (y, x), func = problems.generate_sine_curve_datasets(100, noise=noise)
y_train, x_train = y[: 80], x[: 80]
y_test, x_test = y[80: ], x[80: ]

# Prepare model
# learner = ScikitLearnLearner(x_train, y_train)
learner = GPyTorchLearner(x_train, y_train)
# learner = PyTorchLearner(x_train, y_train)

# Train
# losses = learner.train(num_epochs=5000, learning_rate=0.01)  # for PyTorchLearner
losses = learner.train(num_epochs=200)  # for GPyTorchLearner

# Predict
(y_pred, confidence) = learner.predict(x_test)
(y_pred_train, confidence_train) = learner.predict(x_train)

x_range = numpy.linspace(0, 5.0, 101)  #XXX
(y_range, confidence_range) = learner.predict(x_range)

# Make plots
plots.make_graph("artifacts/graph_test.png", x_range, y_range, confidence=confidence_range, observations=(x_test, y_test), func=func, title=f"{learner.__class__.__name__}")
plots.make_graph("artifacts/graph_train.png", x_range, y_range, confidence=confidence_range, observations=(x_train, y_train), func=func, title=f"{learner.__class__.__name__}")

if confidence_range is not None:
    plots.make_confidence("artifacts/confidence.png", x_range, confidence_range, title=f"{learner.__class__.__name__}")

plots.make_comparison("artifacts/comparison_test.png", y_test, y_pred, confidence, title=f"{learner.__class__.__name__}")
plots.make_comparison("artifacts/comparison_train.png", y_train, y_pred_train, confidence_train, title=f"{learner.__class__.__name__}")

if losses is not None:
    plots.make_loss_history("artifacts/loss.png", losses, title=f"{learner.__class__.__name__}")
