# OpinionDiffusion.jl

## Installation
- [Install Julia](https://julialang.org/downloads/)
- clone repo from github
- Start REPL in your local repository by writing julia
- Press ] to enter package manager
- run following to activate virtual environment and install all required packages
- [More about virtual environments](https://julialang.github.io/Pkg.jl/v1.5/environments/)
```
(v1.6) pkg> activate .
(OpinionDiffusion) pkg> instantiate
```
- press backspace to exit package manager
## Execution
### VS code:
- [VS code and Julia guide](https://www.julia-vscode.org/docs/dev/gettingstarted/#Installation-and-Configuration-1)

### Jupyter notebook:
- [Follow this guide for IJulia](https://github.com/JuliaLang/IJulia.jl)
- everything needed will be installed automatically.
- in Julia REPL write following to start the server:
```
using IJulia
notebook(dir=@__DIR__)
```
- jupyter server should have started and now you can open opinion_diffusion.ipynb

### Pluto notebook:
- [Pluto git repo](https://github.com/fonsp/Pluto.jl)
- Installation :
``` 
julia> ]
(v1.6) pkg> add Pluto PlutoUI
```
- backspace to exit package manager
```
julia> import Pluto
julia> Pluto.run()
```

- when the server starts in your browser open pluto_ntb.jl in pluto web interface.