# parity-nn

Minimal codebase for conducting experiments with neural networks trained on sparse parity learning tasks. With the right selection of hyperparameters, the networks exhibit delayed generalization, i.e. they suddenly generalize past the point of ~0 train loss (a phenomenon often called grokking). If interested in reading more, please take a look at our paper: "A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks".

## Command Reference

To train a fully connected neural net on a parity task:
```shell
python parity.py --train --ind-norms --global-sparsity --subnetworks --faithfulness
```
