@kwdef struct MDS_dim_reduction_config <: Abstract_dim_reduction_config
	out_dim::Int64
end
name(config::MDS_dim_reduction_config) = "MDS"

function reduce_dims(opinions::Matrix{Float64}, dim_reduction_config::MDS_dim_reduction_config)
	model = MultivariateStats.fit(MultivariateStats.MDS, opinions; maxoutdim=dim_reduction_config.out_dim, distances=false)
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