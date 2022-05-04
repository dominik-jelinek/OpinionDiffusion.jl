# voter configs __________________________________________________________________________
abstract type Abstract_voter_init_config end

# diffusion configs ________________________________________________________________________
abstract type Abstract_voter_diff_config end
abstract type Abstract_graph_diff_config end
@kwdef struct Diffusion_config
    checkpoint::Int64
    evolve_vertices::Float64
    evolve_edges::Float64
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