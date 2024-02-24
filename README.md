# One Self-Configurable Model to Solve Many Abstract Visual Reasoning Problems

This repository is the official implementation of the paper titled: *"One Self-Configurable Model to Solve Many Abstract
Visual Reasoning Problems"*.

## Requirements

The recommended way to run the experiments is to use [docker](https://www.docker.com/).
The docker image can be built by running:

```bash
$ docker build -t "mikomel/sal:latest" -f Dockerfile .
```

Alternatively, the requirements can be installed with `pip`:

```bash
$ pip install -r requirements.txt
```

## Tests

Unit tests can be run with:
```bash
PYTHONPATH=. pytest
```

## Datasets

The datasets can be obtained from the corresponding GitHub repositories:

- [G-set](https://github.com/deepiq/deepiq)
- [I-RAVEN](https://github.com/husheng12345/SRAN)
- [PGM](https://github.com/deepmind/abstract-reasoning-matrices)
- [VAP](https://github.com/deepmind/abstract-reasoning-matrices)
- [O3](https://github.com/deepiq/deepiq)

## Training

An exemplary experiment that trains SCAR on I-RAVEN with STL can be run from the project's root directory with:

```bash
# docker
$ ./scripts/train.sh +experiment=stl_raven

# local
$ PYTHONPATH=. python avr/experiment/train.py +experiment=stl_raven
```

For the list of all commands required to reproduce experiments from the paper refer to: `experiment_log.md`

## Repository

The repository contains the following directories:

* `avr/` - Python module that contains the implementation related to data, models, and experiments.
* `config/` - directory with [Hydra](https://github.com/facebookresearch/hydra) configuration files.
* `scripts/` - a set of helpful shell scripts.

## Citation

If you find this work valuable in your research, please cite it as:
```bibtex
@inproceedings{malkinski2024one,
    title={One Self-Configurable Model to Solve Many Abstract Visual Reasoning Problems},
    author={Ma{\l}ki{\'n}ski, Miko{\l}aj and Ma{\'n}dziuk Jacek},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={38},
    year={2024},
}
```
