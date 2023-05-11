module OpinionDiffusion

using Base.Threads
using Base: Float64
using Random
using Dates
using Colors
import Base.@kwdef

using MetaGraphs
using Graphs

import Plots
#import StatsPlots
#import GraphPlot
import KernelDensity
using Makie, GraphMakie, CairoMakie

import Distances
import Clustering
import TSne
using MultivariateStats

import Statistics
import StatsBase
import Distributions
import GaussianMixtures

using JLD2
#import YAML


export parse_data
export init_model
export Logger
export run!
export save_log, load_log, load_logs
export get_opinion, get_vote, get_votes, get_distance, get_ID
export get_voters, get_social_network, get_candidates
export election_summary

export General_model_config, General_graph_diff_config, Diffusion_config
export BA_graph_config, DEG_graph_config

export Spearman_voter_init_config, Spearman_voter_diff_config
export Kendall_voter_init_config, Kendall_voter_diff_config

export reduce_dims, MDS_dim_reduction_config, Tsne_dim_reduction_config, PCA_dim_reduction_config
export clustering, Kmeans_clustering_config, GM_clustering_config, Party_clustering_config, DBSCAN_clustering_config, Density_clustering_config

export plurality_voting, borda_voting, copeland_voting, get_positions
export clustering, draw_voter_vis, draw_heat_vis, unify_projections!, gather_metrics
export gather_vis, timestamp_vis
export get_edge_distances, draw_degree_distr, draw_edge_distances
export draw_range!, draw_voting_res
export voters, social_network
export load_model, restart_model, save_ensemble
export test_KT
export to_string

Bucket = Set{Int64}
Vote = Vector{Bucket}
abstract type Config end

include("parsing.jl")
include("utils.jl")

include("voters/Abstract_voter.jl")
include("voters/Kendall_voter.jl")
include("voters/Spearman_voter.jl")

include("models/Abstract_model.jl")
include("models/General_model.jl")

include("voting_rules.jl")
include("dim_reduction.jl")
include("clustering.jl")

include("graphs/graph.jl")
include("graphs/barabasi_albert.jl")
include("graphs/DEG.jl")

include("Logger.jl")
include("visualization.jl")

end