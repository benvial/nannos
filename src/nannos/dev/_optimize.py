#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import nlopt
import numpy as npo
from scipy.optimize import minimize

from . import backend as bk
from . import grad
from .utils import apply_filter


def simp(x, eps_min, eps_max, p=1):
    return (eps_max - eps_min) * x ** p + eps_min


def project(x, beta=1, thres=0.5):
    x = bk.array(x)

    def tanh(z):
        return bk.tanh(bk.array(z))

    return ((tanh(thres * beta)) + tanh(beta * (x - thres))) / (
        tanh(thres * beta) + (tanh((1 - thres) * beta))
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
        x0 = npo.array(self.x0)
        for iopt in range(*self.threshold):
            print(f"global iteration {iopt}")
            print("-----------------------")

            proj_level = 2 ** iopt
            args = list(self.args)
            args[0] = proj_level
            args = tuple(args)
            if self.method == "scipy":

                def fun_scipy(x, *args):
                    x = bk.array(x, dtype=bk.float64)
                    y = self.fun(x, *args)
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
                    x = bk.array(x, dtype=bk.float64)
                    y = self.fun(x, *args)
                    if gradn.size > 0:
                        dy = self.grad_fun(x, *args)
                        gradn[:] = npo.array(dy, dtype=npo.float64)
                    print(f"current value = {y}")
                    if self.callback is not None:
                        self.callback(x, y, *args)
                    out = npo.float(y)
                    return out

                lb = npo.zeros(self.nvar, dtype=npo.float64)
                ub = npo.ones(self.nvar, dtype=npo.float64)

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
