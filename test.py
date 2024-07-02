#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import problems
import plots
from learner import GPyTorchLearner, ScikitLearnLearner


# y, x = problems.generate_linear_datasets(100)
# y, x = problems.generate_sigmoid_datasets(100)
y, x = problems.generate_sine_curve_datasets(100)
y_train, x_train = y[: 80], x[: 80]
y_test, x_test = y[80: ], x[80: ]

learner = ScikitLearnLearner(x_train, y_train)
# learner = GPyTorchLearner(x_train, y_train)
learner.train()
(y_pred, confidence) = learner.predict(x_test)

plots.make_graph("artifacts/graph.png", x_test, y_test, y_pred, confidence, title=f"{learner.__class__.__name__}")
plots.make_comparison("artifacts/comparison.png", y_test, y_pred, confidence, title=f"{learner.__class__.__name__}")
