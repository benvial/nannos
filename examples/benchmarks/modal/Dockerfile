FROM condaforge/mambaforge:latest

ENV TINI_SUBREAPER="1"
ENV GPU="1"

SHELL ["bash", "-c"]

RUN apt-get update -q && apt-get install -q -y --no-install-recommends \
    libgl1-mesa-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda config --set always_yes yes --set changeps1 no \
    && wget https://gitlab.com/nannos/nannos/-/raw/master/environment-gpu.yml -q \
    && sed -i 's/name: nannos/name: base/g' environment-gpu.yml \
    && mamba env update -n base -f environment-gpu.yml \
    && rm -f environment.yml \
    && conda clean --tarballs --index-cache --packages \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete