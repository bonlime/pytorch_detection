FROM nvcr.io/nvidia/pytorch:20.03-py3

# base setup
RUN apt-get update --fix-missing && \
    apt-get install --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    wget software-properties-common pkg-config build-essential \
    libglu1-mesa -y && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

# Set working directory

# ENV PYTHONPATH "${PYTHONPATH}:/workspace"

# my pip install 
RUN pip --no-cache-dir install --upgrade \
    ConfigArgParse \
    git+https://github.com/bonlime/pytorch-tools.git@v0.1.4 \
    loguru

# Necessary pip packages
# RUN pip install --upgrade pip

# COPY requirements.txt .
# RUN pip install -r requirements.txt
# RUN python3 -m pip install pycocotools==2.0.0

# Copy SSD code
# COPY ./setup.py .
# COPY ./csrc ./csrc
# RUN pip install .

WORKDIR /workspace

# COPY . .
# USER zakirov
# COPY . /workdir
# WORKDIR /workdir