# Adaptive Proximal Algorithms

This repository contains Julia code for the paper
[Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient](https://arxiv.org/abs/2301.04431).

Algorithms are implemented [here](./adaptive_proximal_algorithms.jl).

You can download the datasets required in some of the experiments by running:

```
julia --project=. download_datasets.jl
```

Experiments on a few different problems are contained in subfolders.
For example, run the lasso experiments as follows:

```
julia --project=. lasso/runme.jl
```

This will generate plots in the same subfolder.
