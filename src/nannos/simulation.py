#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

__all__ = ["Simulation"]

from . import backend as bk
from . import jit, logger
from .formulations import fft
from .layers import _get_layer
from .plot import plot_structure, pyvista
from .utils import block, block2list, get_block, inv2by2block, norm, set_index
from .utils import time as timer
from .utils import unique
from .utils.helpers import _reseter

# from .parallel import parloop

_inv = bk.linalg.inv


class Simulation:
    """Main simulation object.

    Parameters
    ----------
    layers : list
        A list of :class:`~nannos.Layer` objects.
    excitation : :class:`~nannos.PlaneWave`
        A plane wave excitation .
    nh : int
        Number of Fourier harmonics (the default is ``100``).
    formulation : str
        Formulation type.  (the default is ``'original'``).
        Available formulations are ``'original'``, ``'tangent'``, ``'jones'`` and ``'pol'``.

    """

    def __init__(
        self,
        layers,
        excitation,
        nh=100,
        formulation="original",
    ):
        # Layers
        self.layers = layers
        self.layer_names = [layer.name for layer in self.layers]
        if not unique(self.layer_names):
            raise ValueError("Layers must have different names")
        # check if all layers share the same lattice
        lattice0 = self.layers[0].lattice
        for layer in self.layers:
            assert layer.lattice == lattice0, ValueError(
                "lattice must be the same for all layers"
            )
        self.lattice = lattice0
        # nh0 is the number of harmonics required as input that might be different
        # from the one used after truncation
        self.nh0 = int(nh)
        nh0 = int(nh)
        # this is to avoid error when truncating to a single harmonic for example
        # when all layers are uniform
        if bk.all(bk.array([s.is_uniform for s in self.layers])) and nh0 != 1:
            nh0 = 1
            logger.info("All layers are uniform, setting nh=1")
        if nh0 == 1:
            nh0 = 2
        # Check formulation
        if formulation not in ["original", "tangent", "jones", "pol"]:
            raise ValueError(
                f"Unknown formulation {formulation}. Please choose between 'original', 'tangent', 'jones' or 'pol'"
            )
        if self.lattice.is_1D and formulation not in ["original", "tangent"]:
            raise ValueError(
                f"Formulation {formulation} not available for 1D gratings. Please choose between 'original' and 'tangent'."
            )
        self.formulation = formulation

        # Get the harmonics
        self.harmonics, self.nh = self.lattice.get_harmonics(nh0)
        # Check if nh and resolution satisfy Nyquist criteria
        maxN = bk.max(self.harmonics)
        if self.lattice.discretization[0] <= 2 * maxN or (
            self.lattice.discretization[1] <= 2 * maxN and not self.lattice.is_1D
        ):
            raise ValueError(f"lattice discretization must be > {2*maxN}")

        # Set the excitation (plane wave)
        self.excitation = excitation
        self.omega = 2 * bk.pi * self.excitation.frequency_scaled
        self.incident_flux = bk.real(
            bk.cos(self.excitation.theta)
            / bk.sqrt(self.layers[0].epsilon * self.layers[0].mu)
        )

        # Buid lattice vectors
        self.k0para = (
            bk.array(self.excitation.wavevector[:2])
            * (self.layers[0].epsilon * self.layers[0].mu) ** 0.5
        )
        r = self.lattice.reciprocal
        self.kx = (
            self.k0para[0]
            + r[0, 0] * self.harmonics[0, :]
            + r[0, 1] * self.harmonics[1, :]
        )
        self.ky = (
            self.k0para[1]
            + r[0, 1] * self.harmonics[0, :]
            + r[1, 1] * self.harmonics[1, :]
        )
        self.Kx = bk.diag(self.kx)
        self.Ky = bk.diag(self.ky)

        # Some useful matrices
        self.IdG = bk.array(bk.eye(self.nh, dtype=bk.complex128))
        self.ZeroG = bk.array(bk.zeros_like(self.IdG, dtype=bk.complex128))

        # Initialize amplitudes
        self.a0 = []
        for i in range(self.nh * 2):
            if i == 0:
                a0 = self.excitation.a0[0]
            elif i == self.nh:
                a0 = self.excitation.a0[1]
            else:
                a0 = 0
            self.a0.append(a0)
        self.a0 = bk.array(self.a0, dtype=bk.complex128)
        self.bN = bk.array(bk.zeros(2 * self.nh, dtype=bk.complex128))

        # This is a boolean checking that the eigenproblems are solved for all layers
        # TODO: This is to avoid solving again, but could be confusing
        self.is_solved = False
        # dictionary to store intermediate S-matrices
        # TODO: check memory consumption of doing that, maybe make it optional.
        self._intermediate_S = dict()

    def get_layer(self, id):
        """Helper to get layer index and name.

        Parameters
        ----------
        id : int or str
            The index of the layer or its name.

        Returns
        -------
        layer : nannos.Layer
            The layer object.
        index : nannos.Layer
            The layer index in the stack.
        """
        return _get_layer(id, self.layers, self.layer_names)

    def get_layer_by_name(self, id):
        return self.get_layer(id)[0]

    def solve(self):
        """Solve the grating problem."""
        _t0 = timer.tic()
        logger.info("Solving")
        layers_solved = []
        for layer in self.layers:
            _t0lay = timer.tic()
            logger.info("Computing eigenpairs for layer {layer}")
            layer = self.build_matrix(layer)
            if layer.is_uniform:
                layer.solve_uniform(self.omega, self.kx, self.ky, self.nh)
            else:
                layer.solve_eigenproblem(layer.matrix)

                # ## normalize
                # phi = layer.eigenvectors
                # out = bk.conj(layer.eigenvectors).T @ layer.Qeps @ layer.eigenvectors
                # layer.eigenvectors /= bk.diag(out)**0.5

            layers_solved.append(layer)

            _t1lay = timer.toc(_t0lay, verbose=False)
            logger.info(
                f"Done computing eigenpairs for layer {layer} in {_t1lay:0.3e}s"
            )
        self.layers = layers_solved
        self.is_solved = True
        _t1 = timer.toc(_t0, verbose=False)
        logger.info(f"Done solving in {_t1:0.3e}s")

    def reset(self, param="all"):
        if param == "S":
            _reseter(self, "S")
            self._intermediate_S = {}
        if param == "solve":
            self.is_solved = False
        if param == "all":
            self.reset("S")
            self.reset("solve")

    def get_S_matrix(self, indices=None):
        """Compute the scattering matrix.

        Parameters
        ----------
        indices : list
            Indices ``[i_first,i_last]`` of the first and last layer (the default is ``None``).
            By default get the S matrix of the full stack.

        Returns
        -------
        list
            The scattering matrix ``[[S11, S12], [S21,S22]]``.

        """

        if not self.is_solved:
            self.solve()

        _t0 = timer.tic()

        S11 = bk.array(bk.eye(2 * self.nh, dtype=bk.complex128))
        S12 = bk.zeros_like(S11)
        S21 = bk.zeros_like(S11)
        S22 = bk.array(bk.eye(2 * self.nh, dtype=bk.complex128))

        if indices is None:
            n_interfaces = len(self.layers) - 1
            stack = range(n_interfaces)
            logger.info("Computing total S-matrix")
        else:
            stack = range(indices[0], indices[1])
            logger.info(f"Computing S-matrix for indices {indices}")

        try:
            S = self._intermediate_S[f"{stack[0]},{stack[-1]}"]
        except Exception:

            for i in stack:
                layer, layer_next = self.layers[i], self.layers[i + 1]
                z = layer.thickness or 0
                z_next = layer_next.thickness or 0
                f, f_next = bk.diag(phasor(layer.eigenvalues, z)), bk.diag(
                    phasor(layer_next.eigenvalues, z_next)
                )
                I_ = _build_Imatrix(layer, layer_next)
                Imat = [
                    [get_block(I_, i, j, 2 * self.nh) for j in range(2)]
                    for i in range(2)
                ]
                A = Imat[0][0] - f @ S12 @ Imat[1][0]

                S11 = bk.linalg.solve(A, f @ S11)
                S12 = bk.linalg.solve(A, (f @ S12 @ Imat[1][1] - Imat[0][1]) @ f_next)
                # B = _inv(A)
                # S11 = B @ f @ S11
                # S12 = B @ ((f @ S12 @ Imat[1][1] - Imat[0][1]) @ f_next)
                S21 = S22 @ Imat[1][0] @ S11 + S21
                S22 = S22 @ Imat[1][0] @ S12 + S22 @ Imat[1][1] @ f_next
                S = [[S11, S12], [S21, S22]]
                self._intermediate_S[f"{stack[0]},{i}"] = S
        if indices is None:
            self.S = S

        _t1 = timer.toc(_t0, verbose=False)
        logger.info(f"Done computing S-matrix {_t1:0.3e}s")
        return S

    def get_z_poynting_flux(self, layer, an, bn):
        if not self.is_solved:
            self.solve()
        q, phi = layer.eigenvalues, bk.array(layer.eigenvectors)
        A = layer.Qeps @ phi @ bk.diag(1 / (self.omega * q))
        phia, phib = phi @ an, phi @ bn
        Aa, Ab = A @ an, A @ bn
        cross_term = 0.5 * (bk.conj(phib) * Aa - bk.conj(Ab) * phia)
        forward_xy = bk.real(bk.conj(Aa) * phia) + cross_term
        backward_xy = -bk.real(bk.conj(Ab) * phib) + bk.conj(cross_term)
        forward = forward_xy[: self.nh] + forward_xy[self.nh :]
        backward = backward_xy[: self.nh] + backward_xy[self.nh :]
        return bk.real(forward), bk.real(backward)

    def get_field_fourier(self, layer_index, z=0):
        layer, layer_index = self.get_layer(layer_index)
        _t0 = timer.tic()
        logger.info(f"Retrieving fields in k-space for layer {layer}")
        if not self.is_solved:
            self.solve()
        ai0, bi0 = self._get_amplitudes(layer_index, translate=False)
        Z = z if hasattr(z, "__len__") else [z]
        fields = bk.zeros((len(Z), 2, 3, self.nh), dtype=bk.complex128)
        for iz, z_ in enumerate(Z):
            ai, bi = _translate_amplitudes(layer, z_, ai0, bi0)
            ht_fourier = layer.eigenvectors @ (ai + bi)
            hx, hy = ht_fourier[: self.nh], ht_fourier[self.nh :]
            A = (ai - bi) / (self.omega * layer.eigenvalues)
            B = layer.eigenvectors @ A
            et_fourier = layer.Qeps @ B
            ey, ex = -et_fourier[: self.nh], et_fourier[self.nh :]
            hz = (self.kx * ey - self.ky * ex) / self.omega
            ez = (self.ky * hx - self.kx * hy) / self.omega
            if layer.is_uniform:
                ez = ez / layer.epsilon
            else:
                # ez = layer.eps_hat_inv @ ez
                ez = bk.linalg.solve(layer.eps_hat, ez)
            for i, comp in enumerate([ex, ey, ez]):
                fields = set_index(fields, [iz, 0, i], comp)
            for i, comp in enumerate([hx, hy, hz]):
                fields = set_index(fields, [iz, 1, i], comp)
        fields_fourier = bk.array(fields)
        _t1 = timer.toc(_t0, verbose=False)
        logger.info(
            f"Done retrieving fields in k-space for layer {layer} in {_t1:0.3e}s"
        )
        return fields_fourier

    def get_ifft(self, u, shape, axes=(0, 1)):
        u = bk.array(u)
        s = 0
        for i in range(self.nh):
            f = bk.zeros(shape, dtype=bk.complex128)
            f = set_index(f, [self.harmonics[0, i], self.harmonics[1, i]], 1.0)
            a = u[i]
            s += a * f
        ft = fft.inverse_fourier_transform(s, axes=axes)
        return ft

    def get_ifft_amplitudes(self, amplitudes, shape, axes=(0, 1)):
        _t0 = timer.tic()
        logger.info("Inverse Fourier transforming amplitudes")

        amplitudes = bk.array(amplitudes)
        if len(amplitudes.shape) == 1:
            amplitudes = bk.reshape(amplitudes, amplitudes.shape + (1,))

        s = 0
        for i in range(self.nh):
            f = bk.zeros(shape + (amplitudes.shape[0],), dtype=bk.complex128)
            f = set_index(f, [self.harmonics[0, i], self.harmonics[1, i]], 1.0)
            # f[self.harmonics[0, i], self.harmonics[1, i], :] = 1.0
            a = amplitudes[:, i]
            s += a * f

        ft = fft.inverse_fourier_transform(s, axes=axes)
        _t1 = timer.toc(_t0, verbose=False)
        logger.info(f"Done inverse Fourier transforming amplitudes in {_t1:0.3e}s")
        return ft

    def get_field_grid(
        self, layer_index, z=0, shape=None, field="all", component="all"
    ):

        if field not in ["all", "E", "H"]:
            raise ValueError("Wrong field argument, must be `all`, `E` or `H`")
        if component not in ["all", "x", "y", "z"]:
            raise ValueError("Wrong component argument, must be `all`, `x`, `y` or `z`")

        layer, layer_index = self.get_layer(layer_index)
        _t0 = timer.tic()
        logger.info("Retrieving fields in real-space for layer {layer}")
        shape = shape or layer.epsilon.shape

        fields_fourier = self.get_field_fourier(layer_index, z)

        fe = fields_fourier[:, 0]
        fh = fields_fourier[:, 1]

        def _get_field(f, i):
            return self.get_ifft_amplitudes(f[:, i, :], shape)

        if component == "all":
            if field == "all":
                E = bk.stack([_get_field(fe, i) for i in range(3)])
                H = bk.stack([_get_field(fh, i) for i in range(3)])
                out = E, H
            elif field == "H":
                out = bk.stack([_get_field(fh, i) for i in range(3)])
            elif field == "E":
                out = bk.stack([_get_field(fe, i) for i in range(3)])
        elif component == "x":
            if field == "all":
                E = _get_field(fe, 0)
                H = _get_field(fh, 0)
                out = E, H
            elif field == "H":
                out = _get_field(fh, 0)
            elif field == "E":
                out = _get_field(fe, 0)
        elif component == "y":
            if field == "all":
                E = _get_field(fe, 1)
                H = _get_field(fh, 1)
                out = E, H
            elif field == "H":
                out = _get_field(fh, 1)
            elif field == "E":
                out = _get_field(fe, 1)
        elif component == "z":
            if field == "all":
                E = _get_field(fe, 2)
                H = _get_field(fh, 2)
                out = E, H
            elif field == "H":
                out = _get_field(fh, 2)
            elif field == "E":
                out = _get_field(fe, 2)

        _t1 = timer.toc(_t0, verbose=False)
        logger.info(
            f"Done retrieving fields in real space for layer {layer} in {_t1:0.3e}s"
        )
        return out

    def get_Efield_grid(self, layer_index, z=0, shape=None, component="all"):
        return self.get_field_grid(
            layer_index, z=z, shape=shape, field="E", component=component
        )

    def get_Hfield_grid(self, layer_index, z=0, shape=None, component="all"):
        return self.get_field_grid(
            layer_index, z=z, shape=shape, field="H", component=component
        )

    def diffraction_efficiencies(self, orders=False, complex=False):
        """Compute the diffraction efficiencies.

        Parameters
        ----------
        orders : bool
            If ``True``, returns diffracted orders, else returns the sum of
            reflection and transmission for all propagating orders (the default is ``False``).

        Returns
        -------
        tuple
            The reflection and transmission ``R`` and ``T``.

        """
        if not hasattr(self, "S"):
            self.get_S_matrix()

        if complex:
            R, T = self._get_complex_orders()
            if not orders:
                R = bk.sum(R, axis=1)
                T = bk.sum(T, axis=1)
        else:
            aN, b0 = self._solve_ext()
            fwd_in, bwd_in = self.get_z_poynting_flux(self.layers[0], self.a0, b0)
            fwd_out, bwd_out = self.get_z_poynting_flux(self.layers[-1], aN, self.bN)

            R = -bwd_in / self.incident_flux
            T = fwd_out / self.incident_flux
            if not orders:
                R = bk.sum(R)
                T = bk.sum(T)
        return R, T

    def _get_complex_orders(self):
        nin = (self.layers[0].epsilon * self.layers[0].mu) ** 0.5
        nout = (self.layers[-1].epsilon * self.layers[-1].mu) ** 0.5
        gamma_in0 = (
            self.omega**2 * nin**2 - self.k0para[0] ** 2 - self.k0para[1] ** 2
        ) ** 0.5
        # gamma_out0 = (
        #     self.omega**2 * nout**2 - self.k0para[0] ** 2 - self.k0para[1] ** 2
        # ) ** 0.5
        gamma_in = (self.omega**2 * nin**2 - self.kx**2 - self.ky**2) ** 0.5
        gamma_out = (self.omega**2 * nout**2 - self.kx**2 - self.ky**2) ** 0.5
        norma_t2 = nin**2 * (gamma_out / gamma_in0)
        norma_r2 = nin**2 * (gamma_in / gamma_in0)

        norma_t = (norma_t2.real) ** 0.5
        norma_r = (norma_r2.real) ** 0.5

        t = self.get_field_fourier("Substrate")[0, 0] * norma_t
        bx0 = self.get_field_fourier("Superstrate")[0, 0] * norma_r
        o0 = bk.zeros(self.nh)
        o0 = set_index(o0, [0], 1)
        r = bk.stack([b - o0 * c for b, c in zip(bx0, self.excitation.amplitude)])
        return r, t

    def get_z_stress_tensor_integral(self, layer_index, z=0):
        layer, layer_index = self.get_layer(layer_index)
        fields_fourier = self.get_field_fourier(layer_index, z=z)
        e = fields_fourier[0, 0]
        h = fields_fourier[0, 1]
        ex = e[0]
        ey = e[1]
        ez = e[2]
        hx = h[0]
        hy = h[1]
        hz = h[2]
        dz = (self.ky * hx - self.kx * hy) / self.omega
        if layer.is_uniform:
            dx = ex * layer.epsilon
            dy = ey * layer.epsilon
        else:
            exy = bk.hstack((-ey, ex))

            # FIXME: anisotropy here?
            # dxy = layer.eps_hat @ exy
            _eps_hat = block([[layer.eps_hat, self.ZeroG], [self.ZeroG, layer.eps_hat]])
            dxy = _eps_hat @ exy
            dx = dxy[self.nh :]
            dy = -dxy[: self.nh]

        Tx = bk.sum(ex * bk.conj(dz) + hx * bk.conj(hz), axis=-1).real
        Ty = bk.sum(ey * bk.conj(dz) + hy * bk.conj(hz), axis=-1).real
        Tz = (
            0.5
            * bk.sum(
                ez * bk.conj(dz)
                + hz * bk.conj(hz)
                - ex * bk.conj(dx)
                - ey * bk.conj(dy)
                - hx * bk.conj(hx)
                - hy * bk.conj(hy),
                axis=-1,
            ).real
        )
        return Tx, Ty, Tz

    def get_order_index(self, order):
        try:
            len(order) == 2
        except Exception:
            if self.lattice.is_1D and isinstance(order, int):
                order = (order, 0)
            else:
                raise ValueError(
                    "order must be a tuple of integers length 2 for bi-periodic gratings"
                )
        return [
            k for k, i in enumerate(self.harmonics.T) if bk.allclose(i, bk.array(order))
        ][0]

    def get_order(self, A, order):
        return A[self.get_order_index(order)]

    def _get_toeplitz_matrix(self, u, transverse=False):
        if transverse:
            return [
                [self._get_toeplitz_matrix(u[i, j]) for j in range(2)] for i in range(2)
            ]
        else:
            uft = fft.fourier_transform(u)
            ix = bk.arange(self.nh)
            jx, jy = bk.meshgrid(ix, ix, indexing="ij")
            delta = self.harmonics[:, jx] - self.harmonics[:, jy]
            return uft[delta[0, :], delta[1, :]]

    def build_matrix(self, layer):
        _t0 = timer.tic()
        layer, layer_index = self.get_layer(layer)
        logger.info(f"Building matrix for layer {layer}")
        if layer.iscopy:
            layer.matrix = layer.original.matrix
            layer.Kmu = layer.original.Kmu
            layer.eps_hat = layer.original.eps_hat
            layer.eps_hat_inv = layer.original.eps_hat_inv
            layer.mu_hat = layer.original.mu_hat
            layer.mu_hat_inv = layer.original.mu_hat_inv
            layer.Keps = layer.original.Keps
            layer.Peps = layer.original.Peps
            layer.Qeps = layer.original.Qeps
            return layer
        Kx, Ky = self.Kx, self.Ky
        if layer.is_uniform:
            epsilon = layer.epsilon
            mu = layer.mu
        else:
            epsilon = layer.epsilon
            mu = layer.mu
        if layer.is_uniform:
            # if layer.is_epsilon_anisotropic:
            #     _epsilon = block(
            #         [
            #             [epsilon[0, 0] * self.IdG, epsilon[0, 1] * self.IdG],
            #             [epsilon[1, 0] * self.IdG, epsilon[1, 1] * self.IdG],
            #         ]
            #     )
            # else:
            #     _epsilon = epsilon
            #
            # if layer.is_mu_anisotropic:
            #     _mu = block(
            #         [
            #             [mu[0, 0] * self.IdG, mu[0, 1] * self.IdG],
            #             [mu[1, 0] * self.IdG, mu[1, 1] * self.IdG],
            #         ]
            #     )
            # else:
            #     _mu = mu

            Keps = _build_Kmatrix(1 / epsilon * self.IdG, Ky, -Kx)
            # Pmu = bk.eye(self.nh * 2)
            Pmu = block([[mu * self.IdG, self.ZeroG], [self.ZeroG, mu * self.IdG]])

        else:
            epsilon_zz = epsilon[2, 2] if layer.is_epsilon_anisotropic else epsilon
            # mu_zz = mu[2, 2] if layer.is_mu_anisotropic else mu
            # TODO: check if mu or epsilon is homogeneous, no need to compute the Toepliz matrix

            eps_hat = self._get_toeplitz_matrix(epsilon_zz)
            # eps_hat = self._get_toeplitz_matrix(epsilon_zz,ana=False)
            mu_hat = self.IdG  # self._get_toeplitz_matrix(mu_zz)
            eps_hat_inv = _inv(eps_hat)
            mu_hat_inv = _inv(mu_hat)
            Keps = _build_Kmatrix(eps_hat_inv, Ky, -Kx)
            Kmu = _build_Kmatrix(mu_hat_inv, Kx, Ky)

            if layer.is_epsilon_anisotropic:
                eps_para_hat = self._get_toeplitz_matrix(epsilon, transverse=True)
            else:
                eps_para_hat = [[eps_hat, self.ZeroG], [self.ZeroG, eps_hat]]
            if self.formulation == "original":
                Peps = block(eps_para_hat)
            elif self.formulation == "tangent":
                if self.lattice.is_1D:
                    N, nuhat_inv = self._get_nu_hat_inv(layer)
                    eps_para_hat = [[eps_hat, self.ZeroG], [self.ZeroG, nuhat_inv]]
                    Peps = block(eps_para_hat)
                else:
                    t = layer.get_tangent_field(self.harmonics, normalize=True)
                    Peps = self._get_Peps(layer, eps_para_hat, t, direct=False)
            elif self.formulation == "jones":
                t = layer.get_tangent_field(self.harmonics, normalize=False)
                J = layer.get_jones_field(t)
                Peps = self._get_Peps(layer, eps_para_hat, J, direct=False)
            else:
                # elif self.formulation == "pol":
                t = layer.get_tangent_field(self.harmonics, normalize=False)
                Peps = self._get_Peps(layer, eps_para_hat, t, direct=False)

            if layer.is_mu_anisotropic:
                mu_para_hat = [
                    [self._get_toeplitz_matrix(mu[i, j]) for j in range(2)]
                    for i in range(2)
                ]

                Pmu = block(mu_para_hat)
            else:
                Pmu = block([[mu_hat, self.ZeroG], [self.ZeroG, mu_hat]])

            # Qeps = self.omega ** 2 * bk.eye(self.nh * 2) - Keps
            # matrix = Peps @ Qeps - Kmu
            matrix = self.omega**2 * Peps @ Pmu - (Peps @ Keps + Kmu @ Pmu)

            layer.matrix = matrix
            layer.Kmu = Kmu
            layer.Pmu = Pmu
            layer.eps_hat = eps_hat
            layer.eps_hat_inv = eps_hat_inv
            layer.mu_hat = mu_hat
            layer.mu_hat_inv = mu_hat_inv
            layer.Keps = Keps
            layer.Peps = Peps

        Qeps = self.omega**2 * Pmu - Keps
        layer.Qeps = Qeps

        _t1 = timer.toc(_t0, verbose=False)
        logger.info(f"Done building matrix for layer {layer} in {_t1:0.3e}s")

        return layer

    def get_epsilon(self, layer_id, axes=(0, 1)):
        # TODO: check formulation and anisotropy

        layer, layer_index = self.get_layer(layer_id)
        if layer.is_uniform:
            return layer.epsilon

        epsilon_zz = (
            layer.epsilon[2, 2] if layer.is_epsilon_anisotropic else layer.epsilon
        )
        eps_hat = self._get_toeplitz_matrix(epsilon_zz)
        out = self.get_ifft(eps_hat[:, 0], shape=self.lattice.discretization, axes=axes)
        return out
        # if inv:
        #     u = layer.eps_hat_inv ## nu_hat
        #     out = self.get_ifft(u[0,:], shape=self.lattice.discretization, axes=axes)
        #     return _inv(out)

    def _get_amplitudes(self, layer_index, z=0, translate=True):
        _t0 = timer.tic()
        logger.info("Retrieving amplitudes")

        layer, layer_index = self.get_layer(layer_index)
        n_interfaces = len(self.layers) - 1
        if layer_index == 0:
            aN, b0 = self._solve_ext()
            ai, bi = self.a0, b0
        elif layer_index == n_interfaces or layer_index == -1:
            aN, b0 = self._solve_ext()
            ai, bi = aN, self.bN
        else:
            ai, bi = self._solve_int(layer_index)
        if translate:
            ai, bi = _translate_amplitudes(self.layers[layer_index], z, ai, bi)
        _t1 = timer.toc(_t0, verbose=False)
        logger.info(f"Done retrieving amplitudes in {_t1:0.3e}s")
        return ai, bi

    def _solve_int(self, layer_index):
        layer, layer_index = self.get_layer(layer_index)
        n_interfaces = len(self.layers) - 1
        S = self.get_S_matrix(indices=(0, layer_index))
        P = self.get_S_matrix(indices=(layer_index, n_interfaces))
        # q = _inv(bk.array(bk.eye(self.nh * 2)) - bk.matmul(S[0][1], P[1][0]))
        # ai = bk.matmul(q, bk.matmul(S[0][0], self.a0))
        Q = bk.array(bk.eye(self.nh * 2)) - bk.matmul(S[0][1], P[1][0])
        ai = bk.linalg.solve(Q, bk.matmul(S[0][0], self.a0))
        bi = bk.matmul(P[1][0], ai)
        return ai, bi

    def _solve_ext(self):
        if not hasattr(self, "S"):
            self.get_S_matrix()
        aN = self.S[0][0] @ self.a0 + self.S[0][1] @ self.bN
        b0 = self.S[1][0] @ self.a0 + self.S[1][1] @ self.bN
        return aN, b0

    def _get_nu_hat_inv(self, layer):
        epsilon = layer.epsilon
        if layer.is_epsilon_anisotropic:
            N = epsilon.shape[2]
            eb = block(epsilon[:2, :2])
            nu = inv2by2block(eb, N)
            # nu = _inv(epsilon[2,2])
            nu1 = bk.array(block2list(nu, N))
            nuhat = self._get_toeplitz_matrix(nu1, transverse=True)
            nuhat_inv = block2list(_inv(block(nuhat)), self.nh)

        else:
            N = epsilon.shape[0]
            nuhat = self._get_toeplitz_matrix(1 / epsilon)
            nuhat_inv = _inv((nuhat))
        return N, nuhat_inv

    def _get_Peps(self, layer, eps_hat, t, direct=False):
        N, nuhat_inv = self._get_nu_hat_inv(layer)

        if direct:

            T = block([[t[1], bk.conj(t[0])], [-t[0], bk.conj(t[1])]])

            invT = inv2by2block(T, N)
            # invT = _inv(T)
            if layer.is_epsilon_anisotropic:
                Q = block(
                    [[eps_hat[0][0], nuhat_inv[0][1]], [eps_hat[1][0], nuhat_inv[1][1]]]
                )

            else:
                Q = block([[eps_hat[0][0], self.ZeroG], [self.ZeroG, nuhat_inv]])

            That = block(
                [
                    [self._get_toeplitz_matrix(get_block(T, i, j, N)) for j in range(2)]
                    for i in range(2)
                ]
            )
            invThat = block(
                [
                    [
                        self._get_toeplitz_matrix(get_block(invT, i, j, N))
                        for j in range(2)
                    ]
                    for i in range(2)
                ]
            )
            Peps = That @ Q @ invThat

        else:
            norm_t = norm(t)
            nt2 = bk.abs(norm_t) ** 2
            Pxx = t[0] * bk.conj(t[0]) / nt2
            Pyy = t[1] * bk.conj(t[1]) / nt2
            Pxy = t[0] * bk.conj(t[1]) / nt2
            Pyx = t[1] * bk.conj(t[0]) / nt2
            Pxx_hat = self._get_toeplitz_matrix(Pxx)
            Pyy_hat = self._get_toeplitz_matrix(Pyy)
            Pxy_hat = self._get_toeplitz_matrix(Pxy)
            Pyx_hat = self._get_toeplitz_matrix(Pyx)

            Pi = block([[Pyy_hat, Pyx_hat], [Pxy_hat, Pxx_hat]])

            # Delta = epsilon - 1 / epsilon
            # Delta_hat = self._get_toeplitz_matrix(Delta)

            eps_para_hat = block(eps_hat)
            if layer.is_epsilon_anisotropic:
                D = eps_para_hat - block(nuhat_inv)
            else:
                D = eps_para_hat - block(
                    [[nuhat_inv, self.ZeroG], [self.ZeroG, nuhat_inv]]
                )

            Peps = eps_para_hat - D @ Pi
        return Peps

    def plot_structure(
        self, plotter=None, nper=(1, 1), dz=0.0, null_thickness=None, **kwargs
    ):
        return plot_structure(self, plotter, nper, dz, null_thickness, **kwargs)


def phasor(q, z):
    return bk.exp(1j * q * z)


# def _build_Kmatrix(u, Kx, Ky):
#     return block(
#         [
#             [Kx @ u @ Kx, Kx @ u @ Ky],
#             [Ky @ u @ Kx, Ky @ u @ Ky],
#         ]
#     )


def _build_Kmatrix(u, Kx, Ky):
    def matmuldiag(A, B):
        return bk.einsum("i,ik->ik", bk.array(bk.diag(A)), bk.array(B))

    kxu = matmuldiag(Kx, u)
    kyu = matmuldiag(Ky, u)
    return block(
        [
            [matmuldiag(Kx.T, kxu.T).T, matmuldiag(Ky.T, kxu.T).T],
            [matmuldiag(Kx.T, kyu.T).T, matmuldiag(Ky.T, kyu.T).T],
        ]
    )


def _build_Mmatrix(layer):
    phi = layer.eigenvectors

    def matmuldiag(A, B):
        return bk.einsum("ik,k->ik", bk.array(A), B)

    # a = layer.Qeps @ phi @ (bk.diag(1 / layer.eigenvalues))
    a = layer.Qeps @ matmuldiag(phi, 1 / layer.eigenvalues)
    return block([[a, -a], [phi, phi]])


# TODO: check orthogonality of eigenvectors to compute M^-1 without inverting it for potential speedup
# cf: D. M. Whittaker and Imat. S. Culshaw, Scattering-matrix treatment of
# patterned multilayer photonic structures
# PHYSICAL REVIEW B, VOLUME 60, NUMBER 4, 1999
#
def _build_Mmatrix_inverse(layer):
    phi = layer.eigenvectors

    def matmuldiag(A, B):
        return bk.einsum("ik,k->ik", bk.array(A), B)

    # b = matmuldiag(phiT,layer.eigenvalues)
    b = bk.diag(layer.eigenvalues) @ bk.conj(phi.T)
    c = bk.conj(phi.T) @ layer.Qeps
    return 0.5 * block([[b, c], [-b, c]])


def _build_Imatrix(layer1, layer2):
    # if layer1.is_uniform:
    #     a1 = _build_Mmatrix(layer1)
    #     inv_a1 = _inv(a1)
    # else:
    #     inv_a1 = _build_Mmatrix_inverse(layer1)
    a1 = _build_Mmatrix(layer1)
    a2 = _build_Mmatrix(layer2)
    return bk.linalg.solve(a1, a2)
    # inv_a1 = _inv(a1)
    # return inv_a1 @ a2


def _translate_amplitudes(layer, z, ai, bi):
    q = layer.eigenvalues
    aim = ai * phasor(q, z)
    bim = bi * phasor(q, layer.thickness - z)
    return aim, bim
