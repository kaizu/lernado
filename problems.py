#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import numpy
import numpy.random


def sine_problem(x, amplitude=1.0, period=numpy.pi * 2, loc=0.0):
    return amplitude * numpy.sin(x * numpy.pi * 2 / period) + loc

def sigmoid_problem(x, gain=1.0, scale=1.0, loc=0.0):
    return scale / (1.0 + numpy.exp(-gain * (x - loc)))

def linear_problem(x, weight=1.0, bias=0.0):
    return weight * x + bias

def white_noise(x, scale=1.0):
    return x + numpy.random.normal(scale=1.0, size=len(x))

def generate_linear_datasets(n):
    x = numpy.random.uniform(0.0, 5.0, size=n)
    y = linear_problem(x, weight=5.0, bias=2.0) + white_noise(x, scale=1.0)
    return (y, x)

def generate_sigmoid_datasets(n):
    x = numpy.random.uniform(0.0, 5.0, size=n)
    y = sigmoid_problem(x, gain=4.0, scale=5.0, loc=2.0) + white_noise(x, scale=1.0)
    return (y, x)

def generate_sine_curve_datasets(n):
    x = numpy.random.uniform(0.0, 5.0, size=n)
    y = sine_problem(x, amplitude=5.0, period=3.0, loc=3.0) + white_noise(x, scale=1.0)
    return (y, x)
