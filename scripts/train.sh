#!/usr/bin/env bash

date
echo "Training command: python avr/experiment/train.py" "${@}"
docker run \
  -v ~/Projects/avr-transfer-learning:/app \
  -v ~/Datasets:/app/data:ro \
  -v ~/.torch:/app/models \
  --rm \
  --ipc host \
  --gpus all \
  --entrypoint python \
  mikomel/sal:latest \
  "avr/experiment/train.py" "${@}"
date
