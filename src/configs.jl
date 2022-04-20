#voter configs __________________________________________________________________________
abstract type Abstract_voter_config end

@kwdef struct Spearman_voter_config <: Abstract_voter_config
    weight_func::Function
    openmindedness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
    stubbornness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
end

abstract type Abstract_voter_diff_config end
@kwdef struct Spearman_voter_diff_config <: Abstract_voter_diff_config
    evolve_vertices::Float64
	attract_proba::Float64
	change_rate::Float64
    normalize_shifts::Union{Nothing, Tuple{Bool, Float64, Float64}}
end

#model configs __________________________________________________________________________
@kwdef struct General_model_config
    m::Integer
    popularity_ratio::Real
    voter_config::Abstract_voter_config
end

abstract type Abstract_graph_diff_config end

@kwdef struct General_graph_diff_config <: Abstract_graph_diff_config
    evolve_edges::Float64
    dist_metric::Distances.Metric
    edge_diff_func::Function
end

@kwdef struct Diffusion_config
    checkpoint::Int64
    voter_diff_config::Abstract_voter_diff_config
    graph_diff_config::Abstract_graph_diff_config
end

# visualizations ________________________________________________________________________
@kwdef struct Kmeans_config
    cluster_count::Int64
end

@kwdef struct GM_config
    cluster_count::Int64
end

@kwdef struct Clustering_config
    used::Bool
    method::String
    kmeans_config::Union{Nothing, Kmeans_config}
    gm_config::Union{Nothing, GM_config}
end

@kwdef struct PCA_config
    out_dim::Int64
end

@kwdef struct Tsne_config
    out_dim::Int64
    reduce_dims::Int64
    max_iter::Int64
    perplexity::Float64
end

@kwdef struct Reduce_dim_config
    method::String
    pca_config::Union{Nothing, PCA_config}
    tsne_config::Union{Nothing, Tsne_config}
end

@kwdef struct Voter_vis_config
    used::Bool
    reduce_dim_config::Union{Nothing, Reduce_dim_config}
    clustering_config::Union{Nothing, Clustering_config}
end