abstract type Abstract_dim_reduction_config <: Config end

function reduce_dims(voters::Vector{Abstract_voter}, dim_reduction_config::T) where T<:Abstract_dim_reduction_config
    opinions = reduce(hcat, get_opinion(voters))

    return reduce_dims(opinions, dim_reduction_config::PCA_dim_reduction_config)
end

@kwdef struct PCA_dim_reduction_config <: Abstract_dim_reduction_config
    out_dim::Int64
end

function reduce_dims(opinions::Matrix{Float64}, dim_reduction_config::PCA_dim_reduction_config)
    model = MultivariateStats.fit(MultivariateStats.PCA, opinions; maxoutdim=dim_reduction_config.out_dim)
    projection = MultivariateStats.transform(model, opinions)
    
    return projection
end

@kwdef struct Tsne_dim_reduction_config <: Abstract_dim_reduction_config
    out_dim::Int64
    reduce_dims::Int64
    max_iter::Int64
    perplexity::Float64
end

function reduce_dims(sampled_opinions, dim_reduction_config::Tsne_dim_reduction_config)
    opinions = permutedims(sampled_opinions)
    
    projection = TSne.tsne(
                        opinions, 
                        dim_reduction_config.out_dim, 
                        dim_reduction_config.reduce_dims, 
                        dim_reduction_config.max_iter, 
                        dim_reduction_config.perplexity
                    )
    
    projection = permutedims(projection)
    
    return projection
end

@kwdef struct MDS_dim_reduction_config <: Abstract_dim_reduction_config
    out_dim::Int64
end

function reduce_dims(sampled_opinions, dim_reduction_config::MDS_dim_reduction_config)
    model = MultivariateStats.fit(MultivariateStats.MDS, sampled_opinions; maxoutdim=dim_reduction_config.out_dim)
    projection = MultivariateStats.transform(model, sampled_opinions)
    
    return projection
end

function unify_projections!(projections, x_projections, y_projections, x_treshold=1.0, y_treshold=1.0)
    if sum(projections[1, 1:length(x_projections)] - x_projections) < x_treshold
		@views row = projections[1, :] 
		row .*= -1.0
	end
	if sum(projections[2, 1:length(y_projections)] - y_projections) < y_treshold
		@views row = projections[2, :] 
		row .*= -1.0
	end
end