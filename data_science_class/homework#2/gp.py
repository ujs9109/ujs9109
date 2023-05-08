import gpytorch
import matplotlib
import pandas as pd
import torch

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, LinearKernel
from gpytorch.constraints import Interval

# 2021-28863 유재상

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        # TODO design your own kernel

        RBF_Q_kernel = (RBFKernel() * RBFKernel())
        Lin_R_kernel = ScaleKernel(LinearKernel() + ScaleKernel(RBFKernel()))

        self.covar_module = ScaleKernel(RBFKernel()) + \
                            ScaleKernel(LinearKernel() + PeriodicKernel(period_length_constraint = Interval(4,11)) + (RBF_Q_kernel)) +\
                            ScaleKernel(LinearKernel() + PeriodicKernel(period_length_constraint=Interval(12, 37)) + ScaleKernel(RBF_Q_kernel)) + \
                            ScaleKernel(LinearKernel() + PeriodicKernel(period_length_constraint=Interval(8, 12)) + ScaleKernel(RBF_Q_kernel)) + \
                            ScaleKernel(LinearKernel() * (PeriodicKernel(period_length_constraint=Interval(12, 26)) + ScaleKernel(RBF_Q_kernel)))

        #self.covar_module = ScaleKernel(RBFKernel()) + ScaleKernel(RBFKernel() * PeriodicKernel()) + ScaleKernel(LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main_run(likelihood, model: ExactGPModel, train_x, train_y, test_x, test_y):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 500
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)

        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if i % 10 == 9:
            print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}   noise: {model.likelihood.noise.item():.3f}')
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.plot(train_x.numpy(), train_y.numpy(), 'k')
        ax.plot(test_x.numpy(), test_y.numpy(), 'r')

        observed_pred = likelihood(model(train_x))
        lower, upper = observed_pred.confidence_region()
        ax.plot(train_x.numpy(), observed_pred.mean.numpy(), 'g')
        ax.fill_between(train_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        observed_pred = likelihood(model(test_x))
        lower, upper = observed_pred.confidence_region()
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        ax.legend(['Train', 'Test (True)', 'Train Mean', 'Train Confidence', 'Predicted Mean', 'Predicted Confidence'], loc='upper left')
        plt.savefig(f'gp.pdf')


if __name__ == '__main__':
    data = pd.read_csv('ch4_mm_gl.txt', skip_blank_lines=True, comment='#', header=0, sep='\s+', dtype=np.float32)
    # standardization
    data['average'] -= np.mean(data['average'])
    data['average'] /= np.std(data['average'])

    n, _ = data.shape
    hold_out = 12 * 6  # 5 year
    train_x = torch.linspace(start=1., end=n - hold_out, steps=n - hold_out)
    test_x = torch.linspace(start=n - hold_out + 1, end=n, steps=hold_out)
    train_y = torch.tensor(data['average'].values[:-hold_out])
    test_y = torch.tensor(data['average'].values[n - hold_out:])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    main_run(likelihood, model, train_x, train_y, test_x, test_y)
