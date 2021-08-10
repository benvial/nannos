# __all__ = ["block"]


from . import numpy as np


def next_power_of_2(x):
    return 1 if x == 0 else int(2 ** np.ceil(np.log2(x)))


def norm(v):
    ## avoid division by 0
    eps = np.finfo(float).eps
    return np.sqrt(eps + np.power(v[0], 2) + np.power(v[1], 2))


def filter(x, rfilt):
    if rfilt == 0:
        return x
    else:
        Nx = x.shape[0]
        # First a 1-D  Gaussian
        # t = np.linspace(0, Nx-1, Nx)
        t = np.linspace(-Nx / 2, Nx / 2, Nx)
        bump = np.exp(-(t ** 2) / rfilt ** 2)
        bump /= np.trapz(bump)  # normalize the integral to 1

        # make a 2-D kernel out of it
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
        kernel_ft = np.fft.fft2(kernel, s=x.shape[:2], axes=(0, 1))

        # convolve
        img_ft = np.fft.fft2(x, axes=(0, 1))
        img2_ft = kernel_ft * img_ft
        out = np.real(np.fft.ifft2(img2_ft, axes=(0, 1)))
        return np.fft.fftshift(out)


def block(a):
    l1 = np.hstack([a[0][0], a[0][1]])
    l2 = np.hstack([a[1][0], a[1][1]])
    return np.vstack([l1, l2])


def get_block(M, i, j, n):
    return M[i * n : (i + 1) * n, j * n : (j + 1) * n]
