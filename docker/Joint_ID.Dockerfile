
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# basic installation for docker development 
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

# installation for pytorch development
RUN pip install -U scikit-learn
RUN pip install pandas
RUN pip install matplotlib
RUN pip install torchsummary
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install configargparse          
RUN pip install imageio-ffmpeg
RUN pip install stories
RUN pip install torchsummaryX
RUN pip install tensorboard
RUN pip install tensorboardX
RUN pip install torch-tb-profiler
RUN pip install mmcv-full               # option
# RUN pip install mmcv-lite
RUN pip install timm
# RUN pip install torchinfo               # option
RUN pip install flopco-pytorch 
# RUN pip install attr                    # option
RUN pip install apex
# RUN pip install gdown                   # option
# RUN pip install unzip                   # option
RUN pip install wandb
