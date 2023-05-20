# Adaptive Proximal Algorithms

This repository contains Julia code for the paper

> Latafat, Themelis, Stella, Patrinos, *Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient*, [arXiv:2301.04431](https://arxiv.org/abs/2301.04431) (2023).

Algorithms are implemented [here](./src/AdaProx.jl).

## Running experiments

Run the following from the repository root:

```sh
# set up environment
julia --project=./experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# download datasets for experiments
julia --project=./experiments download_datasets.jl
```

Then run the scripts from the subfolders.
For example, run the lasso experiments as follows:

```sh
julia --project=./experiments experiments/lasso/runme.jl
```

This will generate plots in the same subfolder.
