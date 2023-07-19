function reduce_dims(voters::Vector{Abstract_voter}, dim_reduction_config::Abstract_dim_reduction_config)
	opinions = reduce(hcat, get_opinion(voters))

	return reduce_dims(opinions, dim_reduction_config)
end

function reduce_dims(opinions::Matrix{Float64}, dim_reduction_config::Abstract_dim_reduction_config)
	throw(NotImplementedError("reduce_dims"))
end
name(config::Abstract_dim_reduction_config) = "Missing dim reduction name"

function unify_projections!(old_projections, new_projections, treshold=0.5)
	for i in axes(old_projections, 1)
		if count(x -> x == 0, sign.(new_projections[i, :]) + sign.(old_projections[i, :])) > treshold * size(old_projections, 2)
			@views row = new_projections[i, :]
			row .*= -1.0
		end
    end
end