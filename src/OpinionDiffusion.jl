module OpinionDiffusion
    
using Distances
using LightGraphs
using Plots

using MetaGraphs

using StatsPlots
using Clustering
using GraphPlot
using TSne
using MultivariateStats
import StatsBase.sample
using ScikitLearn
using Distributions

using Dates
using JLD2
import YAML

include("parsing.jl")
include("utils.jl")


include("analysis.jl")
include("voting.jl")
include("clustering.jl")


include("voters/Abstract_voter.jl")
include("voters/Kendall_voter.jl")
include("voters/Spearman_voter.jl")

include("graph.jl")
include("models/Abstract_model.jl")
include("models/Spearman_model.jl")

include("experiment/Experiment.jl")

include("visualization.jl")

export parse_data2
export Spearman_model
export Experiment
export run_experiment!
export visualize_metrics

end