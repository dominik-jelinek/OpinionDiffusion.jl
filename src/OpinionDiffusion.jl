module OpinionDiffusion


using Base: Float64
using Random
using Dates
import Colors


using MetaGraphs
import Distances
using LightGraphs

import Plots


import StatsPlots
import Clustering
import GraphPlot
import TSne
import MultivariateStats
import StatsBase
import Distributions

#import ScikitLearn

using JLD2
import YAML
import Base.@kwdef

include("configs.jl")
using .Configs
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

export Spearman_model_config, Kmeans_config, GM_config, Clustering_config, 
PCA_config, Tsne_config, Reduce_dim_config, Voter_vis_config, Exp_config, Voter_diff_config, Diffusion_config
end