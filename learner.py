#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import numpy

import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPyTorchLearner:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(
            torch.from_numpy(x_train).clone(),
            torch.from_numpy(y_train).clone(),
            likelihood)
        assert self.model.likelihood is likelihood

    def train(self, niter=50):
        x_train = torch.from_numpy(self.x_train).clone()
        y_train = torch.from_numpy(self.y_train).clone()

        self.model.train()
        self.model.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        for i in range(niter):
            optimizer.zero_grad()
            output = self.model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            msg = "Iter {:d}/{:d} - Loss: {:.3f}   lengthscale: {:.3f}   noise: {:.3f}".format(
                i + 1, niter, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item()
                )
            logger.info(msg)
            optimizer.step()

    def predict(self, x_test):
        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad():
            with gpytorch.settings.fast_pred_var():
                prediction = self.model.likelihood(self.model(torch.from_numpy(x_test).clone()))
            lower, upper = prediction.confidence_region()
            lower, upper = lower.numpy(), upper.numpy()
            mean = prediction.mean.numpy()

        confidence = numpy.array([mean - lower, upper - mean])
        y_pred = mean
        return (y_pred, confidence)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class ScikitLearnLearner:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        kernel = 1 * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e+3)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e+1))
        self.gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

    def train(self):
        self.gaussian_process.fit(self.x_train.reshape(-1, 1), self.y_train)

    def predict(self, x_test):
        y_pred, std_prediction = self.gaussian_process.predict(x_test.reshape(-1, 1), return_std=True)
        return (y_pred, numpy.array([std_prediction, std_prediction]))
