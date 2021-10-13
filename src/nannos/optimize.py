#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import nlopt
from autograd import grad
from scipy.optimize import minimize

from . import numpy as np
from .utils import filter


def project(x, beta=1, thres=0.5):
    return ((np.tanh(thres * beta)) + np.tanh(beta * (x - thres))) / (
        np.tanh(thres * beta) + (np.tanh((1 - thres) * beta))
    )


class StopFunError(Exception):
    """Raised when stop value reached"""

    pass


class StopFun:
    def __init__(self, fun, stopval=None):
        self.fun_in = fun
        self.stopval = stopval

    def fun(self, x, *args):
        self.f = self.fun_in(x, *args)
        if self.stopval is not None:
            if self.f < self.stopval:
                raise StopFunError("Stop value reached.")
        self.x = x
        return self.f


class TopologyOptimizer:
    def __init__(
        self,
        fun,
        x0,
        method="scipy",
        threshold=(0, 8),
        maxiter=10,
        stopval=None,
        args=None,
        callback=None,
        options={},
    ):
        self.fun = fun
        self.x0 = x0
        self.nvar = len(x0)
        self.method = method
        self.threshold = threshold
        self.maxiter = maxiter
        self.stopval = stopval
        self.args = args
        self.callback = callback
        self.options = options
        self.grad_fun = grad(fun)

    def min_function(self):
        f = self.fun(x)

    def minimize(self):

        print("#################################################")
        print(f"Topology optimization with {self.nvar} variables")
        print("#################################################")
        print("")
        x0 = self.x0
        for iopt in range(*self.threshold):
            print(f"global iteration {iopt}")
            print("-----------------------")

            proj_level = 2 ** iopt
            args = list(self.args)
            args[1] = proj_level
            args = tuple(args)
            if self.method == "scipy":

                def fun_scipy(x, *args):
                    y = fun(x, *args)
                    print(f"current value = {y}")
                    if self.callback is not None:
                        self.callback(x, y, *args)
                    return y

                bounds = [(0, 1) for _ in range(self.nvar)]
                options = {"maxiter": self.maxiter}
                if self.options is not None:
                    options.update(self.options)

                stop = StopFun(fun=fun_scipy, stopval=self.stopval)
                try:
                    opt = minimize(
                        stop.fun,
                        x0,
                        args=args,
                        method="L-BFGS-B",
                        jac=self.grad_fun,
                        bounds=bounds,
                        options=options,
                    )
                    xopt = opt.x
                    fopt = opt.fun
                except StopFunError as e:
                    print(e)
                    xopt = stop.x
                    fopt = stop.f

            else:

                def fun_nlopt(x, gradn):
                    gradn[:] = self.grad_fun(x, *args)
                    y = self.fun(x, *args)
                    print(f"current value = {y}")
                    if self.callback is not None:
                        self.callback(x, y, *args)
                    return y

                lb = np.zeros(self.nvar, dtype=float)
                ub = np.ones(self.nvar, dtype=float)

                opt = nlopt.opt(nlopt.LD_MMA, self.nvar)
                opt.set_lower_bounds(lb)
                opt.set_upper_bounds(ub)

                # opt.set_ftol_rel(1e-16)
                # opt.set_xtol_rel(1e-16)
                if self.stopval is not None:
                    opt.set_stopval(self.stopval)
                if self.maxiter is not None:
                    opt.set_maxeval(self.maxiter)

                opt.set_min_objective(fun_nlopt)
                xopt = opt.optimize(x0)
                fopt = opt.last_optimum_value()
            x0 = xopt
        return xopt, fopt
