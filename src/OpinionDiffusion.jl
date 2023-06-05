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
import Statistics
import StatsBase
import Distributions

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

# Voters
export get_opinion, get_vote, get_votes, get_distance, get_ID
export Spearman_voter_init_config, Spearman_voter_diff_config
export Kendall_voter_init_config, Kendall_voter_diff_config

# Graphs
export BA_graph_config, DEG_graph_config

# Model
export get_voters, get_social_network, get_candidates
export init_model
export General_model_config, General_graph_diff_config, Diffusion_config
export run!, run_ensemble_model, run_ensemble
export Logger
export save_log, load_log, load_logs
export load_model, restart_model, save_ensemble

# Diffusion
export SP_init_diff_config, SP_diff_config
export KT_init_diff_config, KT_diff_config
export Graph_init_diff_config, Graph_diff_config
export init_diffusion!

# Visualizations
export get_election_summary, draw_election_summary
export reduce_dims, MDS_dim_reduction_config, Tsne_dim_reduction_config, PCA_dim_reduction_config
export clustering, Kmeans_clustering_config, GM_clustering_config, Party_clustering_config, DBSCAN_clustering_config, Density_clustering_config

export plurality_voting, borda_voting, copeland_voting, get_positions
export clustering, draw_voter_vis, draw_heat_vis, unify_projections!, gather_metrics
export gather_vis, timestamp_vis
export get_edge_distances, draw_degree_distr, draw_edge_distances
export draw_range!, draw_voting_res

# Utils
export parse_data
export to_string
export test_KT

# ______________________________________________________________________________
# Custom Types
# ______________________________________________________________________________

abstract type Config end
abstract type Abstract_voter_init_config <: Config end
abstract type Abstract_graph_init_config <: Config end

abstract type Abstract_diff_config <: Config end
abstract type Abstract_init_diff_config <: Config end

abstract type Abstract_clustering_config <: Config end
abstract type Abstract_dim_reduction_config <: Config end

abstract type Abstract_model_config <: Config end
abstract type Abstract_voter end
abstract type Abstract_model end

Bucket = Set{Int64}
Vote = Vector{Bucket}
@kwdef struct Action
    operation::String
    ID::Union{Int64,Tuple{Int64,Int64}}
    old::Abstract_voter
    new::Abstract_voter
end

include("parsing.jl")
include("utils.jl")

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

include("visualizations/visualization.jl")
include("visualizations/voting_rules.jl")
include("visualizations/dim_reduction.jl")
include("visualizations/clustering.jl")

end