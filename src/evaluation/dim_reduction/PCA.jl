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