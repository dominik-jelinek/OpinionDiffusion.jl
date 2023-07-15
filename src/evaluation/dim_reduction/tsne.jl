@kwdef struct Tsne_dim_reduction_config <: Abstract_dim_reduction_config
	out_dim::Int64
	reduce_dims::Int64
	max_iter::Int64
	perplexity::Float64
end
name(config::Tsne_dim_reduction_config) = "Tsne"

function reduce_dims(opinions::Matrix{Float64}, dim_reduction_config::Tsne_dim_reduction_config)
	opinions = permutedims(opinions)

	projection = TSne.tsne(
		opinions,
		dim_reduction_config.out_dim,
		dim_reduction_config.reduce_dims,
		dim_reduction_config.max_iter,
		dim_reduction_config.perplexity
	)

	return permutedims(projection)
end