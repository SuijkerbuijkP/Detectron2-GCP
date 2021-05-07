FROM nvidia/cuda:10.2-cudnn8-devel

WORKDIR /root
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y libglib2.0-0 && apt-get clean

RUN apt-get update && apt-get install -y \
	ca-certificates git sudo ninja-build curl fuse kmod unzip inotify-tools \
	htop byobu git gcc g++ vim libsm6 libxext6 libxrender-dev lsb-core

# install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# from detectron2 dockerfile
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /home/user

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
# install opencv
RUN pip install opencv-python
# install nightly for D2Go
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
WORKDIR /home/user/
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN python -m pip install --user -e detectron2

# install AdelaiDet
RUN git clone https://github.com/aim-uofa/AdelaiDet.git adet
WORKDIR adet
RUN python setup.py build develop

# install mobileCV
RUN python3 -m pip install 'git+https://github.com/facebookresearch/mobile-vision.git'

# install Detectron2Go
RUN git clone https://github.com/facebookresearch/d2go d2go
WORKDIR d2go
RUN python3 -m pip install .

# Install gcloud.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update
RUN apt-get install -y google-cloud-sdk

# Install cloudml-hypertune for hyperparameter tuning
RUN pip install --upgrade google-cloud-storage
RUN pip install --upgrade cloudml-hypertune gcloud

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# copy in the entrypoint file and trainer folder
WORKDIR /home/user/
RUN mkdir scripts trainer configs model_output
COPY docker scripts
COPY trainer trainer 
COPY configs configs
COPY data data

# set pythonpath for imports 
ENV PYTHONPATH="/home/user/trainer:${PTYHONPATH}"
ENV PYTHONPATH="/home/user/data:${PTYHONPATH}"

ENTRYPOINT ["/home/user/scripts/entrypoint.sh"]
