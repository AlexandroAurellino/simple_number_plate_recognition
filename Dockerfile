# Use the CUDA runtime base image for version 12.6
# FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
#FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
FROM python:3.12

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore

# Install Python 3.10 and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    curl 


# RUN pip install setuptools

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Create working directory
WORKDIR /workspace

# Copy your Python script and requirements file
COPY . /workspace/

# Install Python dependencies
RUN pip install -r requirements.txt

# Tensorflow recommended approach
# RUN python -m pip install tensorflow[and-cuda]

# Verify Python installations
RUN python --version

# Set the default command to run the Python script
CMD ["python", "app.py"]
