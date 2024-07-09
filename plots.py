#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import numpy

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

def make_graph(filename, x_test, y_pred, confidence=None, observations=None, func=None, title=None):
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    if func is not None:
        ax.plot(x_test, func(x_test), 'k--', label="True")
    ax.plot(x_test, y_pred, 'k-', label="Mean prediction")
    if confidence is not None:
        ax.fill_between(
            x_test,
            y_pred - 1. * confidence[0],
            y_pred + 1. * confidence[1],
            alpha=0.5,
            color='silver',
            label=r"Confidence interval",
            )
    if observations is not None:
        ax.plot(observations[0], observations[1], 'k^', label="Observations")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    plt.legend(loc='best')
    if title is not None: ax.set_title(title)
    plt.savefig(filename)
    plt.clf()

def make_loss_history(filename, losses, title=None):
    f, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(numpy.arange(len(losses)), losses, 'k-')

    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Loss')
    if title is not None: ax.set_title(title)
    plt.savefig(filename)
    plt.clf()

def make_confidence(filename, x, confidence, title=None):
    f, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(x, confidence[0] + confidence[1], 'k-')

    ax.set_xlabel('Input')
    ax.set_ylabel('Confidence')
    if title is not None: ax.set_title(title)
    plt.savefig(filename)
    plt.clf()
