
SHELL := /bin/bash


all: pull dld push

pull:
	kaggle kernels pull benjaminvial/nannos -m -p nannos/

push:
	kaggle kernels push -p nannos

dld:
	kaggle kernels output benjaminvial/nannos -p nannos/