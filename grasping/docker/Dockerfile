FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y &&\
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O ./anaconda.sh &&\
  /bin/bash ./anaconda.sh -b -p /opt/conda &&\
  /opt/conda/bin/conda init bash &&\
  rm ./anaconda.sh

RUN mkdir /pkgs
COPY ./environment.yml /pkgs/environment.yml

RUN /opt/conda/bin/conda env create -vv -f /pkgs/environment.yml

RUN mkdir /repos &&\
  cd /repos &&\
  git clone https://github.com/lixiny/manotorch &&\
  cd /repos/manotorch &&\
  /opt/conda/bin/conda run -n py36 pip install --ignore-requires-python -e .

RUN cd /repos &&\
  git clone https://github.com/hassony2/chumpy &&\
  cd /repos/chumpy &&\
  /opt/conda/bin/conda run -n py36 pip install --ignore-requires-python -e .
