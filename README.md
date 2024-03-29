# OpinionDiffusion.jl

## Installation
- Install [Julia](https://julialang.org/)
  - [Juliaup](https://github.com/JuliaLang/juliaup) (recommended)
	- native julia version manager
	- does not require installing Julia first
- clone repo from github
- Start Julia REPL in your local repository by writing julia
- Press ] to enter package manager
```
(v1.9) pkg> dev /path/to/clone/OpinionDiffusion.jl
```
- press backspace to exit package manager
## Execution
### Pluto notebook:
- [Pluto git repo](https://github.com/fonsp/Pluto.jl)
- Installation :
```
julia> ]
(v1.9) pkg> add Pluto PlutoUI
(v1.9) pkg> add Profile, Pkg, Revise, Distributions, Distances, Graphs, Plots, Random, DataFrames, Statistics, GraphMakie, CairoMakie, Colors, JLD2
```
- backspace to exit package manager
```
julia> import Pluto & Pluto.run()
```
- select one of the Pluto notebooks below
#### Pluto_diffusion.jl
- a pluto notebook for interactive creation of a single model

#### Pluto_experiment.jl
- a pluto notebook to run experiments and aggregate statistics

#### Pluto_analysis.jl
- in-depth analysis of saved model states by previous notebooks