FROM mambaorg/micromamba:1.5.8-alpine3.20

WORKDIR /app

# Activate the conda environment during build process
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY ./conda-lock.yml .

# NOTE: `-p` is important to install to the "base" env
RUN micromamba install -y \
    -p /opt/conda \
    -f conda-lock.yml \
  && micromamba clean --all --yes


# Install source
COPY ./pyproject.toml .
COPY ./antarctica_today ./antarctica_today
COPY ./qgis ./qgis

# TODO: `pip install --editable .`
ENV PYTHONPATH=.

# Extend the default micromamba entrypoint to use our CLI
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "antarctica_today"]
