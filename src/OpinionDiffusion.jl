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

include("election/Election.jl")
include("election/parsing/Abstract_parsing.jl")
include("election/parsing/toc.jl")
include("election/parsing/soi.jl")

include("model/voters/Abstract_voter.jl")
include("model/voters/Kendall_voter.jl")
include("model/voters/Spearman_voter.jl")

include("model/graphs/Abstract_graph.jl")
include("model/graphs/barabasi_albert.jl")
include("model/graphs/DEG.jl")
include("model/graphs/random_graph.jl")

include("model/Abstract_model.jl")
include("model/General_model.jl")

include("diffusion/Accumulator.jl")
include("diffusion/Experiment_logger.jl")

include("diffusion/mutations/Abstract_mutation.jl")
include("diffusion/mutations/graph_mutation.jl")
include("diffusion/mutations/kendall_mutation.jl")
include("diffusion/mutations/spearman_mutation.jl")
include("diffusion/diffusion.jl")

include("experiment.jl")
include("ensemble.jl")

include("evaluation/dim_reduction/Abstract_dim_reduction.jl")
include("evaluation/dim_reduction/PCA.jl")
include("evaluation/dim_reduction/tsne.jl")

include("evaluation/clustering/Abstract_clustering.jl")
include("evaluation/clustering/DBSCAN.jl")
include("evaluation/clustering/GM.jl")
include("evaluation/clustering/kmeans.jl")
include("evaluation/clustering/party.jl")
include("evaluation/clustering/watershed.jl")

include("evaluation/voting_rules.jl")
include("evaluation/metrics.jl")
include("evaluation/visualizations.jl")

include("utils.jl")

# ______________________________________________________________________________
# EXPORTS
# ______________________________________________________________________________

# Election
export init_election, Election_config
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
export init_model
export General_model, General_model_config
export get_voters, get_social_network, get_candidates

# Diffusion
export diffusion, diffusion!, init_diffusion!, run_diffusion, run_diffusion!
export Diffusion_config, Diffusion_run_config
export SP_mutation_init_config, SP_mutation_config
export KT_mutation_init_config, KT_mutation_config
export Graph_mutation_init_config, Graph_mutation_config

export run_experiment

# Ensemble
export Ensemble_config, Experiment_config
export ensemble

# logging
export Experiment_logger, init_experiment
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
export get_election_summary, draw_election_summary, draw_election_summary_frequencies
# Dimensionality reduction
export reduce_dims, PCA_dim_reduction_config, Tsne_dim_reduction_config
# Clustering
export clustering, Kmeans_clustering_config, GM_clustering_config, Party_clustering_config, DBSCAN_clustering_config, Watershed_clustering_config

export draw_voter_vis, draw_heat_vis, unify_projections!, gather_metrics
export gather_vis, timestamp_vis
export get_edge_distances, draw_degree_distr, draw_edge_distances
export save_pdf

# Utils
export to_string
export test_KT
export name
end