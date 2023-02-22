#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io
from functools import partial

import jax.numpy as jnp
from jax import custom_vjp

# batched diag
_diag = lambda a: jnp.eye(a.shape[-1]) * a


# batched diagonal, similar to matrix_diag in tensorflow
def _matrix_diag(a):
    reps = jnp.array(a.shape)
    reps = reps.at[:-1].set(1)
    reps = reps.at[-1].set(a.shape[-1])
    # reps[:-1] = 1
    # reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(jnp.tile(a, reps).reshape(newshape))


_dot = partial(jnp.einsum, "...ij,...jk->...ik")


# f :: a -> b
@custom_vjp
def eig(A):
    return jnp.linalg.eig(A)


# f_fwd :: a -> (b, c)
def eig_fwd(A):
    return eig(A), A


# f_bwd :: (c, CT b) -> CT a
def eig_bwd(A, CTb):
    """Gradient of a general square (complex valued) matrix"""
    e, u = CTb  # eigenvalues as 1d array, eigenvectors in columns
    print(e)
    print(u)
    n = e.shape[-1]
    ge, gu = CTb
    ge = _matrix_diag(ge)
    f = 1 / (e[..., jnp.newaxis, :] - e[..., :, jnp.newaxis] + 1.0e-20)
    # print(f)

    f -= _diag(f)
    ut = jnp.swapaxes(u, -1, -2)
    r1 = f * _dot(ut, gu)
    r2 = -f * (_dot(_dot(ut, jnp.conj(u)), jnp.real(_dot(ut, gu)) * jnp.eye(n)))
    r = _dot(_dot(jnp.linalg.inv(ut), ge + r1 + r2), ut)
    # if not jnp.iscomplexobj(CTb):
    #     r = jnp.real(r)
    # the derivative is still complex for real input (imaginary delta is allowed), real output
    # but the derivative should be real in real input case when imaginary delta is forbidden

    return (r,)


eig.defvjp(eig_fwd, eig_bwd)


import numpy as onp
from jax import grad, jacfwd, jacrev

n = 3
A = onp.random.rand(n, n) + 1j * onp.random.rand(n, n)
A = jnp.array(A)
# print(eig(A))
print(jacrev(eig, holomorphic=True)(A))

# print(jacfwd(eig,holomorphic=True)(A))
