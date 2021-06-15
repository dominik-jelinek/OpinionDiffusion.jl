module OpinionDiffusion
    

using Distances: metrics
using MultivariateStats: pairwise, length
using LightGraphs
using Plots
plotlyjs()
using MetaGraphs
using StatsPlots
using Clustering
using Distances
using GraphPlot
using TSne
using MultivariateStats
import StatsBase.sample
using ScikitLearn
using Distributions
using JLD2

using Profile
using BenchmarkTools
using Printf
using Dates
import YAML

include("parsing.jl")
include("utils.jl")


include("analysis.jl")
include("voting.jl")
include("visualization.jl")

include("voters/Abstract_voter.jl")
include("voters/Kendall_voter.jl")
include("voters/Spearman_voter.jl")

include("graph.jl")
include("models/Abstract_model.jl")
include("models/Spearman_model.jl")

include("experiment/Experiment.jl")

export parse_data2
export Spearman_model
export Experiment
end