#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16


def make_comparison(filename, y_test, y_pred, confidence, title=None):
    vmin, vmax = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    vmin, vmax = vmin - (vmax - vmin) * 0.1, vmax + (vmax - vmin) * 0.1

    f, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot((vmin, vmax), (vmin, vmax), 'k--')
    if confidence is not None:
        ax.errorbar(y_test, y_pred, yerr=confidence, fmt='ko')
    else:
        ax.plot(y_test, y_pred, 'ko')

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')
    if title is not None: ax.set_title(title)
    plt.savefig(filename)
    plt.clf()

def make_graph(filename, x_test, y_test, y_pred, confidence, title=None):
    f, ax = plt.subplots(1, 1, figsize=(8, 6))

    if confidence is not None:
        ax.errorbar(x_test, y_pred, yerr=confidence, fmt='k.', label="Prediction")
    else:
        ax.plot(x_test, y_pred, 'ko')
    ax.plot(x_test, y_test, 'r*', label="Observation")

    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    if title is not None: ax.set_title(title)
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()
