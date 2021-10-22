FROM debian:stretch
LABEL maintainer="Contextual Dynamics Laboratory <contextualdynamics@gmail.com>"
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    CONDA_PINNED=/opt/conda/conda-meta/pinned \
    NOTEBOOK_DIR=/mnt \
    NOTEBOOK_IP=0.0.0.0 \
    NOTEBOOK_PORT=8888
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY docker-setup/pin_conda_package_version.sh /etc/profile.d/
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc \
    && apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends eatmydata \
    && eatmydata apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        g++ \
        git \
        procps \
        sudo \
        vim \
        wget \
    && eatmydata apt-get install -y python-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && conda config --set auto_update_conda false \
    && conda config --prepend channels conda-forge \
    && conda config --set channel_priority strict \
    && conda config --set show_channel_urls true \
    && conda install -Sy  \
           conda=4.10.3 \
           ipython=7.27.0 \
           matplotlib \
           "mpi=1.0=mpich" \
           "mpi4py=3.0.3" \
           "mpich=3.3.2" \
           nltk \
           notebook=6.4.4 \
           pandas \
           pandoc=2.10 \
           pip=20.0.2 \
           pyspellchecker \
           "scikit-learn[alldeps]>=0.18" \
           "scipy!=1.0.0" \
           seaborn \
           setuptools=49.6.0 \
           sqlalchemy \
           tini=0.18.0 \
    && conda clean -afy \
    && conda install pytorch==1.7.1 cpuonly -c pytorch \
    && pip install --no-cache-dir \
           numba==0.54.0 \
           hypertools \
           "pydata-wrangler>=0.1.4" \
           quail \
           git+https://github.com/brainiak/brainiak.git@938151acff10cf49954f2c9933278de327b9da9d \
           ipywidgets \
           pycircstat \
           statsmodels \
           tqdm \
           nose \
    && pip install numpy==1.20.0 \
    && source /etc/profile.d/pin_conda_package_version.sh \
    && version_locked_pkgs=( hypertools matplotlib mpi4py mpich nltk numpy \
                             pandas pydata-wrangler pyspellchecker quail \
                             scikit-learn scipy seaborn sqlalchemy ) \
    && for pkg in "${version_locked_pkgs[@]}"; do \
           pin_package "$pkg" exact equal; \
        done \
    && jupyter notebook --generate-config \
    && ipython profile create \
    && sed -i \
        -e 's/^# c.Completer.use_jedi = True/c.Completer.use_jedi = False/' \
        -e 's/^# c.IPCompleter.use_jedi = True/c.IPCompleter.use_jedi = False/' \
        ~/.ipython/profile_default/ipython_config.py
WORKDIR /mnt
ENTRYPOINT ["tini", "-g", "--"]
CMD ["jupyter", "notebook"]
COPY docker-setup/jupyter_notebook_config.py /root/.jupyter/