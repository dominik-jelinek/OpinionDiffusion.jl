module OpinionDiffusion

# ______________________________________________________________________________
# IMPORTS
# ______________________________________________________________________________

# Base
using Base.Threads
using Base: Float64
using Random
using Dates
using Colors
import Base.@kwdef

# Basic
import Distances
using Statistics
import StatsBase
import Distributions
using DataFrames
using PartialFunctions

# Graphs
using Graphs
using MetaGraphs

# Visualizations
import Plots
using Makie, GraphMakie, CairoMakie

# Clustering
import Clustering
import TSne
import GaussianMixtures

# Dimensionality reduction
using MultivariateStats

import KernelDensity

# Serialization
using JLD2

# ______________________________________________________________________________
# EXPORTS
# ______________________________________________________________________________

# election
export parse_data
export select
export Selection_config

# Voters
export init_voters, get_opinion, get_vote, get_votes, get_distance, get_ID
export Spearman_voter_init_config, Spearman_voter_diff_config
export Kendall_voter_init_config, Kendall_voter_diff_config

# Graphs
export init_graph
export BA_graph_config, DEG_graph_config

# Model
export General_model
export get_voters, get_social_network, get_candidates
export init_model
export General_model_config, General_graph_diff_config, Diffusion_config
export run!, run_ensemble_model, run_ensemble
export Logger
export save_log, load_log, load_logs
export load_model, restart_model, save_ensemble

export Experiment_config, Ensemble_config
export ensemble

# Diffusion
export SP_diff_init_config, SP_diff_config
export KT_diff_init_config, KT_diff_config
export Graph_diff_init_config, Graph_diff_config
export init_diffusion!

# Metrics
export extract
export draw_range!, draw_voting_res
export plurality_voting, borda_voting, copeland_voting, get_positions

# Visualizations
export get_election_summary, draw_election_summary
export reduce_dims, MDS_dim_reduction_config, Tsne_dim_reduction_config, PCA_dim_reduction_config
export clustering, Kmeans_clustering_config, GM_clustering_config, Party_clustering_config, DBSCAN_clustering_config, Density_clustering_config

export clustering, draw_voter_vis, draw_heat_vis, unify_projections!, gather_metrics
export gather_vis, timestamp_vis
export get_edge_distances, draw_degree_distr, draw_edge_distances

# Utils
export to_string
export test_KT

# ______________________________________________________________________________
# Custom Types
# ______________________________________________________________________________

abstract type Config end
abstract type Abstract_voter_init_config <: Config end
abstract type Abstract_graph_init_config <: Config end

abstract type Abstract_diff_config <: Config end
abstract type Abstract_diff_init_config <: Config end

abstract type Abstract_clustering_config <: Config end
abstract type Abstract_dim_reduction_config <: Config end

abstract type Abstract_model_config <: Config end
abstract type Abstract_voter end
abstract type Abstract_model end

Bucket = Set{Int64}
Vote = Vector{Bucket}
candidate_count(vote::Vote) = sum([length(bucket) for bucket in vote])

@kwdef struct Action
    operation::String
    ID::Union{Int64,Tuple{Int64,Int64}}
    old::Abstract_voter
    new::Abstract_voter
end

include("election/election.jl")
include("election/toc.jl")
include("election/soi.jl")

include("voters/Abstract_voter.jl")
include("voters/Kendall_voter.jl")
include("voters/Spearman_voter.jl")

include("graphs/graph.jl")
include("graphs/barabasi_albert.jl")
include("graphs/DEG.jl")
include("graphs/random_graph.jl")

include("diffusions/diffusion.jl")
include("diffusions/graph_diffusion.jl")
include("diffusions/kendall_diffusion.jl")
include("diffusions/spearman_diffusion.jl")

include("models/Abstract_model.jl")
include("models/General_model.jl")
include("models/Logger.jl")

include("evaluation/visualizations.jl")
include("evaluation/metrics.jl")
include("evaluation/voting_rules.jl")
include("evaluation/dim_reduction.jl")
include("evaluation/clustering.jl")

include("utils.jl")

end