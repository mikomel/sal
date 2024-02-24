FROM nvcr.io/nvidia/pytorch:23.11-py3

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6  # OpenCV dependencies

RUN mkdir -p /app/datasets
RUN mkdir -p /app/models

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY .env /app/.env
COPY config /app/config
COPY avr /app/avr

WORKDIR /app
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:."
CMD ["/bin/bash"]
