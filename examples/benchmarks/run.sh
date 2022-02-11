#!/bin/bash


module load miniconda
source activate nannos
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=TRUE

for THREADS in 1 2 4 8 16
do
  mkdir -p $THREADS
  pytest -svvv ../../test/benchmarks.py $THREADS
  mv *.npz $THREADS/
done
