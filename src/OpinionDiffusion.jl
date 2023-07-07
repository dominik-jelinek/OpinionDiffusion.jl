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
using StatsBase
using Distributions
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

# KDE
import KernelDensity

# Serialization
using JLD2

# ______________________________________________________________________________
# Custom Types
# ______________________________________________________________________________

abstract type Abstract_config end
abstract type Abstract_voter_config <: Abstract_config end
abstract type Abstract_graph_config <: Abstract_config end

abstract type Abstract_mutation_config <: Abstract_config end
abstract type Abstract_mutation_init_config <: Abstract_config end

abstract type Abstract_clustering_config <: Abstract_config end
abstract type Abstract_dim_reduction_config <: Abstract_config end

abstract type Abstract_model_config <: Abstract_config end

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

include("models/Abstract_model.jl")
include("models/General_model.jl")

include("evaluation/Accumulator.jl")

@kwdef struct Diffusion_run_config
	diffusion_steps::Int64
	mutation_configs::Vector{Abstract_mutation_config}
end

@kwdef struct Diffusion_config <: Abstract_config
	diffusion_init_config::Union{Vector{Abstract_mutation_init_config}, Nothing}
	diffusion_run_config::Diffusion_run_config
end

@kwdef struct Experiment_config
	election_config::Election_config
	model_config::Abstract_model_config
	diffusion_config::Union{Diffusion_config, Nothing} = nothing
end
include("logging/Experiment_logger.jl")


include("diffusion.jl")
include("mutations/graph_mutation.jl")
include("mutations/kendall_mutation.jl")
include("mutations/spearman_mutation.jl")

include("ensemble.jl")

include("evaluation/visualizations.jl")
include("evaluation/metrics.jl")
include("evaluation/voting_rules.jl")
include("evaluation/dim_reduction.jl")
include("evaluation/clustering.jl")

include("utils.jl")

# ______________________________________________________________________________
# EXPORTS
# ______________________________________________________________________________

# Election
export Candidate, Election
export parse_data
export remove_candidates
export Sampling_config, sample

# Voters
export init_voters
export get_vote, get_votes, get_opinion, get_distance, get_ID, get_properties, get_property
export Spearman_voter, Spearman_voter_config
export Kendall_voter, Kendall_voter_config

# Graphs
export init_graph
export BA_graph_config, DEG_graph_config, Random_graph_config

# Model
export General_model
export get_voters, get_social_network, get_candidates

# Diffusion
export Diffusion_config, Diffusion_run_config
export init_diffusion!
export SP_mutation_init_config, SP_mutation_config
export KT_mutation_init_config, KT_mutation_config
export Graph_mutation_init_config, Graph_mutation_config
export run, run!
export run_experiment

# Ensemble
export Ensemble_config, Experiment_config
export ensemble

# logging
export Model_logger, Experiment_logger
export save_model, load_model
export save_config, save_configs, load_config, load_configs
export save_ensemble, load_ensemble

# Metrics
export Accumulator, add_metrics!, accumulated_metrics, get_metrics
export agg_stats, retrieve_variable
export col_name
export compare_metric, compare_metric!, draw_metric!, draw_metric
export compare_voting_rule, compare_voting_rule!, draw_voting_rule, draw_voting_rule!
export plurality_voting, borda_voting, copeland_voting, get_positions

# Visualizations
export get_election_summary, draw_election_summary
export reduce_dims, PCA_dim_reduction_config, Tsne_dim_reduction_config, MDS_dim_reduction_config
export clustering, Kmeans_clustering_config, GM_clustering_config, Party_clustering_config, DBSCAN_clustering_config, Density_clustering_config

export draw_voter_vis, draw_heat_vis, unify_projections!, gather_metrics
export gather_vis, timestamp_vis
export get_edge_distances, draw_degree_distr, draw_edge_distances

# Utils
export to_string
export test_KT
export name
end