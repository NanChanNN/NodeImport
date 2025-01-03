# NodeImport: Imbalanced Node Classification with Node Importance Assessment

This is the code for the KDD 2025 paper of NodeImport: **NodeImport: Imbalanced Node Classification with Node Importance Assessment**.

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

For instance, to run the model on the _Cora_ dataset with a _GCN_ backbone, an imbalance ratio of _50_, for _5_ runs on GPU device _4_, execute:

```sh
./run.sh Cora GCN 50 5 4
```

## Acknowledgments
The implementation of GNN backbones in the `nets` folder is adapted from [TAM](https://github.com/Jaeyun-Song/TAM). We thank the authors of TAM for their valuable contributions.

## Citation
If you find our work useful, please cite:

```
Nan Chen, Zemin Liu, Bryan Hooi, Bingsheng He, Jun Hu, and Jia Chen. 2025. NodeImport: Imbalanced Node Classification with Node Importance Assessment. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD ’25), August 3–7, 2025, Toronto, ON, Canada. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3690624.3709215
```

Feel free to contact nanchansysu@gmail.com if you have any questions.
