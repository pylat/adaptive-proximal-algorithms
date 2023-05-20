# Adaptive Proximal Algorithms

This repository contains Julia code for the paper

> Latafat, Stella, Themelis, Patrinos, *Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient*, [arXiv:2301.04431](https://arxiv.org/abs/2301.04431) (2023).

Algorithms are implemented [here](./src/AdaProx.jl).

## Running experiments

Navigate to the `experiments` folder and run the following:

```
julia --project=. -e "using Pkg; Pkg.instantiate()" # instantiate environment
julia --project=. download_datasets.jl # download datasets for experiments
```

Then run the scripts from the subfolders.
For example, run the lasso experiments as follows:

```
julia --project=. lasso/runme.jl
```

This will generate plots in the same subfolder.
