
SHELL := /bin/bash


pull:
	kaggle kernels pull benjaminvial/nannos_cpu -m -p nannos_gpu/
	kaggle kernels pull benjaminvial/nannos_gpu -m -p nannos_gpu/

push:
	kaggle kernels push -p nannos

dld:
	kaggle kernels output benjaminvial/nannos_cpu -p nannos_cpu/
	kaggle kernels pull benjaminvial/nannos_gpu -m -p nannos_gpu/