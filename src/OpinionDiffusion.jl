module OpinionDiffusion

using Base.Threads
using Base: Float64
using Random
using Dates
import Colors
import Base.@kwdef

using MetaGraphs
using Graphs
using SimpleWeightedGraphs

import Gadfly
import Plots
import StatsPlots
import GraphPlot

import Distances
import Clustering
import TSne
using MultivariateStats

import StatsBase
import Distributions
#import ScikitLearn

using JLD2
import YAML


export parse_data
export General_model
export Logger
export run!
export save_log, load_log, load_logs
export get_opinion, get_vote
export get_votes

export General_model_config, General_graph_diff_config, Diffusion_config
export Spearman_voter_config, Spearman_voter_diff_config
export Kendall_voter_config, Kendall_voter_diff_config

export Kmeans_config, GM_config, Clustering_config, PCA_config, Tsne_config, Reduce_dim_config, Voter_vis_config
export plurality_voting, borda_voting, copeland_voting
export reduce_dim, clustering, draw_voter_vis, draw_heat_vis, unify_projections!
export get_edge_distances, draw_degree_distr, draw_edge_distances
export draw_range, draw_voting_res
export voters, social_network
export load_model, restart_model
export test_KT

include("configs.jl")

include("parsing.jl")
include("utils.jl")

include("voting.jl")
include("clustering.jl")


include("voters/Abstract_voter.jl")
include("voters/Kendall_voter.jl")
include("voters/Spearman_voter.jl")

include("graph.jl")
include("models/Abstract_model.jl")
include("models/General_model.jl")

include("Logger.jl")

include("visualization.jl")

end