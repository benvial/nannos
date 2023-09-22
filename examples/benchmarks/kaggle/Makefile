
SHELL := /bin/bash


pull:
	kaggle kernels pull benjaminvial/nannos-cpu -m -p nannos_cpu/
	kaggle kernels pull benjaminvial/nannos-gpu -m -p nannos_gpu/

push:
	kaggle kernels push -p nannos_cpu
	kaggle kernels push -p nannos_gpu

dld:
	kaggle kernels output benjaminvial/nannos-cpu -p nannos_cpu/ -o
	kaggle kernels output benjaminvial/nannos-gpu -p nannos_gpu/ -o