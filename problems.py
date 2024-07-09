#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import functools
import numpy
import numpy.random


def sine_problem(x, amplitude=1.0, period=numpy.pi * 2, loc=0.0):
    return amplitude * numpy.sin(x * numpy.pi * 2 / period) + loc

def sigmoid_problem(x, gain=1.0, scale=1.0, loc=0.0):
    return scale / (1.0 + numpy.exp(-gain * (x - loc)))

def linear_problem(x, weight=1.0, bias=0.0):
    return weight * x + bias

def white_noise(x, scale=1.0):
    return numpy.random.normal(scale=scale, size=len(x))

def generate_datasets(func, n, noise):
    x = numpy.random.uniform(0.0, 5.0, size=n)
    y = func(x)
    if noise > 0:
        y += white_noise(x, scale=noise)
    return (y, x), func

def generate_linear_datasets(n, noise=1.0):
    func = functools.partial(linear_problem, weight=5.0, bias=2.0)
    return generate_datasets(func, n, noise)

def generate_sigmoid_datasets(n, noise=1.0):
    func = functools.partial(sigmoid_problem, gain=10.0, scale=5.0, loc=2.0)
    return generate_datasets(func, n, noise)

def generate_sine_curve_datasets(n, noise=1.0):
    func = functools.partial(sine_problem, amplitude=5.0, period=3.0, loc=3.0)
    return generate_datasets(func, n, noise)
