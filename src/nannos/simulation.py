#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from . import numpy as np
from .formulations import fft
from .formulations.analytical import fourier_transform_circle
from .formulations.jones import get_jones_field
from .formulations.tangent import get_tangent_field
from .utils import block, get_block, norm

__all__ = ["Simulation"]


class Simulation:
    """Main simulation object.

    Parameters
    ----------
    lattice : :class:`~nannos.Lattice`
        The lattice.
    layers : list
        A list of :class:`~nannos.Layer` objects (the default is ``None``).
    excitation : :class:`~nannos.PlaneWave`
        A plane wave excitation (the default is ``None``).
    nh : int
        Number of Fourier harmonics (the default is ``100``).
    formulation : str
        Formulation type.  (the default is ``'original'``).
        Available formulations are ``'original'``, ``'tangent'`` and ``'jones'``.
    truncation : str
        Truncation method.  (the default is ``'circular'``).
        Available methods are ``'circular'`` and ``'parallelogrammic'``.

    """

    def __init__(
        self,
        lattice,
        layers=[],
        excitation=None,
        nh=100,
        formulation="original",
        truncation="circular",
    ):
        self.lattice = lattice
        self.layers = layers or []
        self.excitation = excitation
        self.nh0 = int(nh)
        if self.nh0 == 1:
            self.nh0 = 2
        self.formulation = formulation
        self.truncation = truncation
        self.harmonics, self.nh = self.lattice.get_harmonics(
            self.nh0, method=self.truncation
        )
        self.omega = 2 * np.pi * self.excitation.frequency
        self.k0para = np.array(self.excitation.wavevector[:2]) * np.sqrt(
            self.layers[0].epsilon * self.layers[0].mu
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
        self.Kx = np.diag(self.kx)
        self.Ky = np.diag(self.ky)
        self.IdG = np.eye(self.nh)
        self.ZeroG = np.zeros_like(self.IdG)

        self.a0 = np.zeros(2 * self.nh, dtype=complex)

        # self.a0[0] = self.excitation.amplitude[0]
        # self.a0[self.nh] = self.excitation.amplitude[1]

        self.a0 = []
        for i in range(self.nh * 2):
            if i == 0:
                a0 = self.excitation.amplitude[0]
            elif i == self.nh:
                a0 = self.excitation.amplitude[1]
            else:
                a0 = 0
            self.a0.append(a0)
        self.a0 = np.array(self.a0)

        self.bN = np.zeros(2 * self.nh, dtype=complex)

        self.is_solved = False

        self._layer_names = [l.name for l in self.layers]
        if np.unique(self._layer_names).size != len(self._layer_names):
            raise ValueError("Layers must have different names")

    def _get_layer(self, id):
        if isinstance(id, str):
            if id in self._layer_names:
                for i, l in enumerate(self.layers):
                    if l.name == id:
                        return l, i
            else:
                raise ValueError(f"Unknown layer name {id}")
        elif isinstance(id, int):
            return self.layers[id], id
        else:
            raise ValueError(
                f"Wrong id for layer: {id}. Please use an integrer specifying the layer index or a string for the layer name"
            )

    def solve(self):
        """Solve the grating problem."""
        layers_solved = []
        for layer in self.layers:
            layer = self._build_matrix(layer)
            # layer.solve_eigenproblem(layer.matrix)
            if layer.is_uniform:
                # layer.eigenvectors = np.eye(self.nh*2)
                layer.solve_uniform(self.omega, self.kx, self.ky, self.nh)
            else:
                layer.solve_eigenproblem(layer.matrix)
            layers_solved.append(layer)
        self.layers = layers_solved
        self.is_solved = True

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

        S11 = np.eye(2 * self.nh, dtype=complex)
        S12 = np.zeros_like(S11)
        S21 = np.zeros_like(S11)
        S22 = np.eye(2 * self.nh, dtype=complex)

        if indices is None:
            n_interfaces = len(self.layers) - 1
            stack = range(n_interfaces)
        else:
            stack = range(indices[0], indices[1])

        for i in stack:
            layer, layer_next = self.layers[i], self.layers[i + 1]
            z = layer.thickness or 0
            z_next = layer_next.thickness or 0
            f, f_next = np.diag(phasor(layer.eigenvalues, z)), np.diag(
                phasor(layer_next.eigenvalues, z_next)
            )
            I_ = _build_Imatrix(layer, layer_next)
            I = [[get_block(I_, i, j, 2 * self.nh) for j in range(2)] for i in range(2)]
            A = I[0][0] - f @ S12 @ I[1][0]
            B = np.linalg.inv(A)
            S11 = B @ f @ S11
            S12 = B @ ((f @ S12 @ I[1][1] - I[0][1]) @ f_next)
            S21 = S22 @ I[1][0] @ S11 + S21
            S22 = S22 @ I[1][0] @ S12 + S22 @ I[1][1] @ f_next
        S = [[S11, S12], [S21, S22]]
        if indices is None:
            self.S = S

        return S

    def get_z_poynting_flux(self, layer, an, bn):
        if not self.is_solved:
            self.solve()
        q, phi = layer.eigenvalues, layer.eigenvectors
        A = (layer.Qeps @ phi) @ np.diag(1 / (self.omega * q))
        pa, pb = phi @ an, phi @ bn
        Aa, Ab = A @ an, A @ bn
        cross_term = 0.5 * (np.conj(pb) * Aa - np.conj(Ab) * pa)
        forward_xy = np.real(np.conj(Aa) * pa) + cross_term
        backward_xy = -np.real(np.conj(Ab) * pb) + np.conj(cross_term)
        forward = forward_xy[: self.nh] + forward_xy[self.nh :]
        backward = backward_xy[: self.nh] + backward_xy[self.nh :]
        return forward, backward

    def get_field_fourier(self, layer_index, z=0):
        if not self.is_solved:
            self.solve()

        layer, layer_index = self._get_layer(layer_index)

        ai0, bi0 = self._get_amplitudes(layer_index, translate=False)

        Z = [z] if np.isscalar(z) else z

        fields = []
        for z_ in Z:
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
                ez = layer.eps_hat_inv @ ez

            fields.append([[ex, ey, ez], [hx, hy, hz]])

        self.fields_fourier = np.array(fields)
        return self.fields_fourier

    def get_ifft_amplitudes(self, amplitudes, shape, axes=(0, 1)):

        amplitudes = np.array(amplitudes)
        # print("amplitudes.shape", amplitudes.shape)
        if len(amplitudes.shape) == 1:
            amplitudes = np.reshape(amplitudes, amplitudes.shape + (1,))

        s = 0
        for i in range(self.nh):
            f = np.zeros(shape + (amplitudes.shape[0],), dtype=complex)
            # print("f.shape", f.shape)
            f[self.harmonics[0, i], self.harmonics[1, i], :] = 1.0
            a = amplitudes[:, i]
            # print("a.shape", a.shape)
            s += a * f

        # print("s.shape", s.shape)
        ft = fft.inverse_fourier_transform(s, axes=axes)
        # print("ft.shape", ft.shape)
        return ft

    def get_field_grid(self, layer_index, z=0, shape=None):
        layer, layer_index = self._get_layer(layer_index)
        shape = shape or layer.patterns[0].epsilon.shape

        fields_fourier = self.get_field_fourier(layer_index, z)

        fe = fields_fourier[:, 0]
        fh = fields_fourier[:, 1]

        E = np.array([self.get_ifft_amplitudes(fe[:, i, :], shape) for i in range(3)])
        H = np.array([self.get_ifft_amplitudes(fh[:, i, :], shape) for i in range(3)])
        return E, H

    def diffraction_efficiencies(self, orders=False):
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
        # if not hasattr(self, "S"):
        # self.get_S_matrix()

        aN, b0 = self._solve_ext()
        fwd_in, bwd_in = self.get_z_poynting_flux(self.layers[0], self.a0, b0)
        fwd_out, bwd_out = self.get_z_poynting_flux(self.layers[-1], aN, self.bN)

        F0 = np.cos(self.excitation.theta) / np.sqrt(
            self.layers[0].epsilon * self.layers[0].mu
        )

        R = np.real(-bwd_in / F0)
        T = np.real(fwd_out / F0)
        if not orders:
            R = np.sum(R)
            T = np.sum(T)
        return R, T

    def get_z_stress_tensor_integral(self, layer_index, z=0):
        if not hasattr(self, "fields_fourier"):
            self.get_field_fourier(layer_index, z=z)
        # ex, ey, ez = self.fields_fourier[0, 0,:,:]
        # hx, hy, hz = self.fields_fourier[0, 1,:,:]

        e = self.fields_fourier[0, 0]
        h = self.fields_fourier[0, 1]
        ex = e[0]
        ey = e[1]
        ez = e[2]

        hx = h[0]
        hy = h[1]
        hz = h[2]

        layer, layer_index = self._get_layer(layer_index)
        dz = (self.ky * hx - self.kx * hy) / self.omega
        if layer.is_uniform:
            dx = ex * layer.epsilon
            dy = ey * layer.epsilon
        else:
            exy = np.hstack((-ey, ex))
            dxy = layer.eps_hat @ exy
            dx = dxy[self.nh :]
            dy = -dxy[: self.nh]

        Tx = np.sum(ex * np.conj(dz) + hx * np.conj(hz), axis=-1).real
        Ty = np.sum(ey * np.conj(dz) + hy * np.conj(hz), axis=-1).real
        Tz = (
            0.5
            * np.sum(
                ez * np.conj(dz)
                + hz * np.conj(hz)
                - ex * np.conj(dx)
                - ey * np.conj(dy)
                - hx * np.conj(hx)
                - hy * np.conj(hy),
                axis=-1,
            ).real
        )
        return Tx, Ty, Tz

    def get_order_index(self, order):
        try:
            len(order) == 2
        except:
            raise ValueError("order must be a tuple of length 2")
        return [k for k, i in enumerate(self.harmonics.T) if np.allclose(i, order)][0]

    def get_order(self, A, order):
        return A[self.get_order_index(order)]

    def _get_toeplitz_matrix(self, u, transverse=False):
        if transverse:
            return [
                [self._get_toeplitz_matrix(u[i, j]) for j in range(2)] for i in range(2)
            ]
        else:
            if self.formulation == "analytical":
                # TODO: analytical FT
                raise NotImplementedError(
                    "Analytical formulation not currently available."
                )
            else:
                uft = fft.fourier_transform(u)
            ix = range(self.nh)
            jx, jy = np.meshgrid(ix, ix, indexing="ij")
            delta = self.harmonics[:, jx] - self.harmonics[:, jy]
            return uft[delta[0, :], delta[1, :]]

    def _build_matrix(self, layer):
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
            epsilon = layer.patterns[0].epsilon
            mu = layer.patterns[0].mu
        is_mu_anisotropic = np.shape(mu)[:2] == (3, 3)
        is_epsilon_anisotropic = np.shape(epsilon)[:2] == (3, 3)
        if layer.is_uniform:
            I = np.eye(self.nh)
            if is_epsilon_anisotropic:
                _epsilon = block(
                    [
                        [epsilon[0, 0] * I, epsilon[0, 1] * I],
                        [epsilon[1, 0] * I, epsilon[1, 1] * I],
                    ]
                )
            else:
                _epsilon = epsilon

            if is_mu_anisotropic:
                _mu = block(
                    [[mu[0, 0] * I, mu[0, 1] * I], [mu[1, 0] * I, mu[1, 1] * I]]
                )
            else:
                _mu = mu
            Keps = _build_Kmatrix(1 / epsilon * self.IdG, Ky, -Kx)
            # Pmu = np.eye(self.nh * 2)
            Pmu = block([[mu * self.IdG, self.ZeroG], [self.ZeroG, mu * self.IdG]])

            Qeps = self.omega ** 2 * Pmu - Keps
        else:
            epsilon_zz = epsilon[2, 2] if is_epsilon_anisotropic else epsilon
            mu_zz = mu[2, 2] if is_mu_anisotropic else mu
            # TODO: check if mu or epsilon is homogeneous, no need to compute the Toepliz matrix

            eps_hat = self._get_toeplitz_matrix(epsilon_zz)
            # eps_hat = self._get_toeplitz_matrix(epsilon_zz,ana=False)
            mu_hat = self.IdG  # self._get_toeplitz_matrix(mu_zz)
            eps_hat_inv = np.linalg.inv(eps_hat)
            mu_hat_inv = np.linalg.inv(mu_hat)
            Keps = _build_Kmatrix(eps_hat_inv, Ky, -Kx)
            Kmu = _build_Kmatrix(mu_hat_inv, Kx, Ky)

            if is_epsilon_anisotropic:
                eps_para_hat = self._get_toeplitz_matrix(epsilon, transverse=True)
            else:
                eps_para_hat = [[eps_hat, self.ZeroG], [self.ZeroG, eps_hat]]

            if self.formulation == "original":
                Peps = block(eps_para_hat)
            elif self.formulation == "jones":
                t = get_tangent_field(epsilon_zz, normalize=True)
                # norm_t = norm(t)
                # t /= norm_t.max()
                J = get_jones_field(t)
                # J /= norm(J).max()
                Peps = self._get_Peps(epsilon, eps_para_hat, J, direct=False)
            elif self.formulation == "tangent":
                t = get_tangent_field(epsilon_zz)
                Peps = self._get_Peps(epsilon, eps_para_hat, t)
            else:
                raise ValueError(
                    f"Unknown formulation {self.formulation}. Please choose between 'original', 'tangent' or 'jones'"
                )

            if is_mu_anisotropic:
                mu_para_hat = [
                    [self._get_toeplitz_matrix(mu[i, j]) for j in range(2)]
                    for i in range(2)
                ]

                Pmu = block(mu_para_hat)
            else:
                Pmu = block([[mu_hat, self.ZeroG], [self.ZeroG, mu_hat]])

            # Qeps = self.omega ** 2 * np.eye(self.nh * 2) - Keps
            # matrix = Peps @ Qeps - Kmu
            Qeps = self.omega ** 2 * Pmu - Keps
            matrix = self.omega ** 2 * Peps @ Pmu - (Peps @ Keps + Kmu @ Pmu)

            layer.matrix = matrix
            layer.Kmu = Kmu
            layer.eps_hat = eps_hat
            layer.eps_hat_inv = eps_hat_inv
            layer.mu_hat = mu_hat
            layer.mu_hat_inv = mu_hat_inv
            layer.Keps = Keps
            layer.Peps = Peps

        layer.Qeps = Qeps

        return layer

    def _get_amplitudes(self, layer_index, z=0, translate=True):
        n_interfaces = len(self.layers) - 1
        if layer_index == 0:
            aN, b0 = self._solve_ext()
            ai, bi = self.a0, b0
        elif layer_index == n_interfaces:
            aN, b0 = self._solve_ext()
            ai, bi = aN, self.bN
        else:
            ai, bi = self._solve_int(layer_index)
            layer = self.layers[layer_index]
        if translate:
            ai, bi = _translate_amplitudes(self.layers[layer_index], z, ai, bi)
        return ai, bi

    def _solve_int(self, layer_index):
        n_interfaces = len(self.layers) - 1
        S = self.get_S_matrix(indices=(0, layer_index))
        P = self.get_S_matrix(indices=(layer_index, n_interfaces))
        q = np.linalg.inv(np.eye(self.nh * 2) - np.dot(S[0][1], P[1][0]))
        ai = np.dot(q, np.dot(S[0][0], self.a0))
        bi = np.dot(P[1][0], ai)
        return ai, bi

    def _solve_ext(self):
        if not hasattr(self, "S"):
            self.get_S_matrix()
        aN = self.S[0][0] @ self.a0 + self.S[0][1] @ self.bN
        b0 = self.S[1][0] @ self.a0 + self.S[1][1] @ self.bN
        return aN, b0

    def _get_Peps(self, epsilon, eps_hat, t, direct=False):

        is_epsilon_anisotropic = np.shape(epsilon)[:2] == (3, 3)
        if is_epsilon_anisotropic:
            N = epsilon.shape[2]
            eb = block(epsilon[:2, :2])
            nu = _inv2by2block(eb, N)
            # nu = np.linalg.inv(epsilon[2,2])
            nu1 = np.array(_block2list(nu, N))
            nuhat = self._get_toeplitz_matrix(nu1, transverse=True)

            nuhat_inv = _block2list(np.linalg.inv(block(nuhat)), self.nh)

        else:
            N = epsilon.shape[0]
            nuhat = self._get_toeplitz_matrix(1 / epsilon)
            nuhat_inv = np.linalg.inv((nuhat))
        if direct:
            T = block([[t[1], np.conj(t[0])], [-t[0], np.conj(t[1])]])
            invT = _inv2by2block(T, N)
            if is_epsilon_anisotropic:
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
            nt2 = np.abs(norm_t) ** 2
            Pxx = t[0] ** 2 / nt2
            Pyy = t[1] ** 2 / nt2
            Pxy = t[0] * np.conj(t[1]) / nt2
            Pyx = t[1] * np.conj(t[0]) / nt2
            Pxx_hat = self._get_toeplitz_matrix(Pxx)
            Pyy_hat = self._get_toeplitz_matrix(Pyy)
            Pxy_hat = self._get_toeplitz_matrix(Pxy)
            Pyx_hat = self._get_toeplitz_matrix(Pyx)

            Pi = block([[Pyy_hat, Pyx_hat], [Pxy_hat, Pxx_hat]])

            # Delta = epsilon - 1 / epsilon
            # Delta_hat = self._get_toeplitz_matrix(Delta)

            eps_para_hat = block(eps_hat)
            if is_epsilon_anisotropic:
                D = eps_para_hat - block(nuhat_inv)
            else:
                D = eps_para_hat - block(
                    [[nuhat_inv, self.ZeroG], [self.ZeroG, nuhat_inv]]
                )

            Peps = eps_para_hat - D @ Pi
        return Peps


def phasor(q, z):
    return np.exp(1j * q * z)


def _build_Kmatrix(u, Kx, Ky):
    return block(
        [
            [Kx @ u @ Kx, Kx @ u @ Ky],
            [Ky @ u @ Kx, Ky @ u @ Ky],
        ]
    )


def _build_Mmatrix(layer):
    phi = layer.eigenvectors
    a = layer.Qeps @ phi @ (np.diag(1 / layer.eigenvalues))
    return block([[a, -a], [phi, phi]])


def _build_Imatrix(layer1, layer2):
    a1 = _build_Mmatrix(layer1)
    a2 = _build_Mmatrix(layer2)
    return np.linalg.inv(a1) @ a2


def _translate_amplitudes(layer, z, ai, bi):
    q = layer.eigenvalues
    aim = ai * phasor(q, z)
    bim = bi * phasor(q, layer.thickness - z)
    return aim, bim


def _block2list(M, N):
    return [[get_block(M, i, j, N) for j in range(2)] for i in range(2)]


def _inv2by2block(T, N):
    M = _block2list(T, N)
    detT = M[0][0] * M[1][1] - M[1][0] * M[0][1]
    return block(
        [
            [M[1][1] / detT, -M[0][1] / detT],
            [-M[1][0] / detT, M[0][0] / detT],
        ]
    )
