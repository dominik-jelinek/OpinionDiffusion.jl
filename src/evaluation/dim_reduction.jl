function reduce_dims(voters::Vector{Abstract_voter}, dim_reduction_config::T) where {T<:Abstract_dim_reduction_config}
    opinions = reduce(hcat, get_opinion(voters))

    return reduce_dims(opinions, dim_reduction_config)
end

@kwdef struct PCA_dim_reduction_config <: Abstract_dim_reduction_config
    out_dim::Int64
end
name(config::PCA_dim_reduction_config) = "PCA"

function reduce_dims(opinions::Matrix{Float64}, dim_reduction_config::PCA_dim_reduction_config)
    model = MultivariateStats.fit(MultivariateStats.PCA, opinions; maxoutdim=dim_reduction_config.out_dim)
    projection = MultivariateStats.transform(model, opinions)
    display(model)
    return projection
end

@kwdef struct Tsne_dim_reduction_config <: Abstract_dim_reduction_config
    out_dim::Int64
    reduce_dims::Int64
    max_iter::Int64
    perplexity::Float64
end
name(config::Tsne_dim_reduction_config) = "Tsne"

function reduce_dims(sampled_opinions::Matrix{Float64}, dim_reduction_config::Tsne_dim_reduction_config)
    opinions = permutedims(sampled_opinions)

    projection = TSne.tsne(
        opinions,
        dim_reduction_config.out_dim,
        dim_reduction_config.reduce_dims,
        dim_reduction_config.max_iter,
        dim_reduction_config.perplexity
    )

    return permutedims(projection)
end

@kwdef struct MDS_dim_reduction_config <: Abstract_dim_reduction_config
    out_dim::Int64
end
name(config::MDS_dim_reduction_config) = "MDS"

function reduce_dims(sampled_opinions::Matrix{Float64}, dim_reduction_config::MDS_dim_reduction_config)
    model = MultivariateStats.fit(MultivariateStats.MDS, sampled_opinions; maxoutdim=dim_reduction_config.out_dim, distances=false)
    projection = MultivariateStats.predict(model)

    return projection
end

function unify_projections!(old_projections, new_projections, treshold=0.5)
    for i in axes(old_projections, 1)
        if count(x -> x == 0, sign.(new_projections[i, :]) + sign.(old_projections[i, :])) > treshold * size(old_projections, 2)
            @views row = new_projections[i, :]
            row .*= -1.0
        end
    end
end