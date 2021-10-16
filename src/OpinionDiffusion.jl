module OpinionDiffusion

using Base: Float64
using Random
using Dates
import Colors
import Base.@kwdef

using MetaGraphs
using LightGraphs

import Plots
import StatsPlots
import GraphPlot

import Distances
import Clustering
import TSne
import MultivariateStats
import StatsBase
import Distributions
#import ScikitLearn

using JLD2
import YAML


export parse_data2
export Spearman_model
export Experiment
export run_experiment!
export visualize_metrics

export Spearman_model_config, Kmeans_config, GM_config, Clustering_config, PCA_config, Tsne_config, Reduce_dim_config, Voter_vis_config, Exp_config, Voter_diff_config, Edge_diff_config, Diffusion_config

include("configs.jl")

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

end