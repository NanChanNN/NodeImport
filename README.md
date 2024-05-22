# NodeInfl: Imbalanced Node Classification with Node Influence Assessment

This repository contains the official implementation of **NodeInfl: Imbalanced Node Classification with Node Influence Assessment**.

## Overview

Key components of the repository:

- **Main Training Procedure**: Implemented in `main.py`.
- **Importance Calculation**: Implemented in `calculate_importance` in `imp_calc.py`.

## Requirements

To install the necessary packages, run:

```setup
pip install -r requirements.txt
```

## Datasets

- We use the built-in datasets from the [PyTorch Geometric package](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).
- Datasets will be downloaded into the default `data` folder upon first execution.

## Running

Hyper-parameter configurations for all Dataset+GNN settings are available in the `run.sh` file. To run the model as described in the paper, specify the required hyper-parameters and run the following command:

```run
./run.sh [dataset] [GNN backbone] [imbalance ratio] [running times] [GPU device]
```

For instance, to run the model on the _Cora_ dataset with a _GCN_ backbone, an imbalance ratio of _50_, for _5_ runs on GPU device 4, execute:

```sh
./run.sh Cora GCN 50 5 4
```

## Acknowledgments
The implementation of GNN backbones in the `nets` folder is adapted from [TAM](https://github.com/Jaeyun-Song/TAM). We thank the authors of TAM for their valuable contributions.
