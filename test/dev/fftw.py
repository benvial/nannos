import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


from timeit import Timer

import numpy
import pyfftw
import scipy

N = 2**11

a = pyfftw.empty_aligned((N, N), dtype="complex128")
b = pyfftw.empty_aligned((N, N), dtype="complex128")

a[:] = numpy.random.randn(N, N) + 1j * numpy.random.randn(N, N)
b[:] = numpy.random.randn(N, N) + 1j * numpy.random.randn(N, N)

t = Timer(lambda: numpy.fft.fft(a))

print("Time with numpy.fftpack: %1.3f seconds" % t.timeit(number=100))
t = Timer(lambda: scipy.fft.fft(a))

print("Time with scipy.fftpack: %1.3f seconds" % t.timeit(number=100))

# Monkey patch fftpack with pyfftw.interfaces.scipy_fftpack
scipy.fftpack = pyfftw.interfaces.scipy_fftpack
scipy.fft.fft(a)  # We cheat a bit by doing the planning first

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

print("Time with monkey patched scipy_fftpack: %1.3f seconds" % t.timeit(number=100))
