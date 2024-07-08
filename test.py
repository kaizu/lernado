#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import problems
import plots
from learner import PyTorchLearner, GPyTorchLearner, ScikitLearnLearner


noise = 0.5
# y, x = problems.generate_linear_datasets(100, noise=noise)
# y, x = problems.generate_sigmoid_datasets(100, noise=noise)
y, x = problems.generate_sine_curve_datasets(100, noise=noise)
y_train, x_train = y[: 80], x[: 80]
y_test, x_test = y[80: ], x[80: ]

# learner = ScikitLearnLearner(x_train, y_train)
# learner = GPyTorchLearner(x_train, y_train)
learner = PyTorchLearner(x_train, y_train)
losses = learner.train(num_epochs=5000, learning_rate=0.01)
(y_pred, confidence) = learner.predict(x_test)
(y_pred_train, confidence_train) = learner.predict(x_train)

plots.make_graph("artifacts/graph_test.png", x_test, y_test, y_pred, confidence, title=f"{learner.__class__.__name__}")
plots.make_comparison("artifacts/comparison_test.png", y_test, y_pred, confidence, title=f"{learner.__class__.__name__}")
plots.make_graph("artifacts/graph_train.png", x_train, y_train, y_pred_train, confidence_train, title=f"{learner.__class__.__name__}")
plots.make_comparison("artifacts/comparison_train.png", y_train, y_pred_train, confidence_train, title=f"{learner.__class__.__name__}")

if losses is not None:
    plots.make_loss_history("artifacts/loss.png", losses, title=f"{learner.__class__.__name__}")
