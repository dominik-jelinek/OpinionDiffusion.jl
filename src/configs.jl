@kwdef struct Spearman_model_config
    weight_func::Function
    m::Integer
    openmindedness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
    stubbornness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
end

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

@kwdef struct Exp_config
    sample_size::Int64
    voter_vis_config::Voter_vis_config
end

@kwdef struct Voter_diff_config
    evolve_vertices::Int64
	attract_proba::Float64
	change_rate::Float64
    method::String
end

@kwdef struct Edge_diff_config
    evolve_edges::Int64
    dist_metric::Distances.Metric
    edge_diff_func::Function
end


@kwdef struct Diffusion_config
    diffusions::Int64
    checkpoint::Int64
    voter_diff_config::Voter_diff_config
    edge_diff_config::Edge_diff_config
end