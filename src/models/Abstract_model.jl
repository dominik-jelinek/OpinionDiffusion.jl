get_voters(model::T) where {T<:Abstract_model} = model.voters
get_social_network(model::T) where {T<:Abstract_model} = model.social_network
get_candidates(model::T) where {T<:Abstract_model} = model.candidates

function select_neighbor(self, model; rng=Random.GLOBAL_RNG)
	voters = get_voters(model)
	social_network = get_social_network(model)
	neighbors_ = neighbors(social_network, get_ID(self))

	if length(neighbors_) == 0
		return nothing
	end

	neighbor_id = neighbors_[rand(rng, 1:end)]
	neighbor = voters[neighbor_id]

	return neighbor
end