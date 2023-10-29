# Start with Ubuntu 23.04 as base image
FROM ubuntu:23.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

# Install necessary dependencies
RUN apt install -y build-essential cmake git libopencl1 ocl-icd-opencl-dev clblast-utils 

# Clone and build Llama.cpp
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp

# Build the project with CLBlast
RUN mkdir build && cd build && cmake .. -DLLAMA_CLBLAST=ON && make
