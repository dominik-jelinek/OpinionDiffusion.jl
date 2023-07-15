function reduce_dims(voters::Vector{Abstract_voter}, dim_reduction_config::Abstract_dim_reduction_config)
	opinions = reduce(hcat, get_opinion(voters))

	return reduce_dims(opinions, dim_reduction_config)
end

function reduce_dims(opinions::Matrix{Float64}, dim_reduction_config::Abstract_dim_reduction_config)
	throw(NotImplementedError("reduce_dims"))
end
name(config::Abstract_dim_reduction_config) = "Missing dim reduction name"