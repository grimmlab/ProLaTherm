FROM nvidia/cuda:11.3.0-base-ubuntu20.04
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip  &&  \
    apt-get install -y vim && apt-get install -y git
RUN mkdir /configfiles
COPY docker_requirements.txt /configfiles
RUN pip3 install -r /configfiles/docker_requirements.txt
