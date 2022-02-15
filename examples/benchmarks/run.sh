#!/bin/bash


# module load miniconda
conda activate nannos
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=TRUE

echo $CONDA_PREFIX
echo $(which python)

for THREADS in 1 2 4 8 16
do
  echo "##################################################"
  echo "############## $THREADS threads ##################"
  echo "##################################################"
  export OMP_NUM_THREADS=$THREADS
  export BLAS_NUM_THREADS=$THREADS
  export OPENBLAS_NUM_THREADS=$THREADS
  export MKL_NUM_THREADS=$THREADS
  export VECLIB_MAXIMUM_THREADS=$THREADS
  export NUMEXPR_NUM_THREADS=$THREADS
  export XLA_FLAGS='--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads='$THREADS
  mkdir -p $THREADS
  # taskset -c 1-$THREADS python benchmarks.py $THREADS $1
  python benchmarks.py $THREADS $1
  mv *.npz $THREADS/
done
