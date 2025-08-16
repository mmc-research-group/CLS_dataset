# CLS_dataset
[![python](https://img.shields.io/badge/python-v3.12.3-blue)](https://www.python.org/downloads/release/python-3123/)

This repository contains the code to generate the dataset used in the paper "[Hamiltionan simulation for nonlinear partial differential equation by Schrödingerization](https://arxiv.org/abs/2508.01640)".
CLS stands for "Carleman linearization + Schrödingerization".
It is a Hamiltionian simulation method for nonlienar partial differential equaiton.
This method is a new approach to time evolution simulation of nonlinear partial differential equations in quantum computing.


## File discreption
This project is organized as follows:
- `config.json`: This file configures the simulation by setting the physical variables, computaitional domain, and time evolution parameters for the numerical solvers.

- `computation_modules.py`: This Python script defines functions to solve a nonlinear reaction-diffusion equation using three different numerical methods: a standard Finite Difference Method (FDM) on CPU, and Carleman Linearization (CL) and Carleman Linearization + Schrödingerization (CLS) on GPU. It saves the simulation results as NumPy arrays and optional VTK files for visualization.

- `simulation_runner.py`: This Python script acts as a master controller for running a series of numerical simulations. It reads parameters from a `config.json` file, sets up the output directories, and then systematically calls different computation functions (FDM, Carleman Linearization, and CLS) from the `computation_modules` module for various parameter combinations.

## Requirements
|  Software  |  Version  |
| :----: | :----: |
|  python  |  3.12.3  |
| numpy | 2.0.0 |
| cupy | 13.2.0 |
| tqdm | 4.66.5 |

## Usage
First, clone this repository to your local machine.
```sh
git clone https://github.com/mmc-research-group/CLS_dataset.git
```
Set the parameters and the path in `config.json`.

- `M_values` in `config.json` means number of computational points for $x$.
- `N_values` in `config.json` means number of computational points for $p$.
  
Other variables in `config.json` correspond to variables in the paper.

Excute `simulation_runner.py`
```sh
python simulation_runner.py
```
The results will be saved in `results/%Y-%m-%d_%H-%M-%S`

## Citation

```sh
Shoya Sasaki, Katsuhiro Endo and Mayu Muramatsu, Hamiltonian simulation for nonlinear partial differential equation by Schrödingerization, arXiv preprint arXiv:2508.01640.
```

In Bibtex format
```sh
@article{sasaki2025hamiltonian,
  title={Hamiltonian simulation for nonlinear partial differential equation by Schr$\backslash$"$\{$o$\}$ dingerization},
  author={Sasaki, Shoya and Endo, Katsuhiro and Muramatsu, Mayu},
  journal={arXiv preprint arXiv:2508.01640},
  year={2025}
}
```

## Lisence
This project is licensed under the MIT LICENSE - see the [LICENSE.txt](LICENSE.txt) file for details

