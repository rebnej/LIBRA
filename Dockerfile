FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive


RUN pip install --upgrade pip

RUN pip install cython

RUN pip install numpy --upgrade


RUN pip install matplotlib --ignore-installed
RUN pip install numpy --ignore-installed
RUN pip install packaging --ignore-installed
RUN pip install pycocotools --ignore-installed
RUN pip install tensorboardX --ignore-installed
RUN pip install tqdm --ignore-installed

RUN pip install nltk


RUN pip install transformers --ignore-installed
RUN pip install sentencepiece

