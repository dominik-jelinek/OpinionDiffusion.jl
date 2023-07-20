get_voters(model::Abstract_model) = model.voters
get_social_network(model::Abstract_model) = model.social_network
get_candidates(model::Abstract_model) = model.candidates

"""
    init_model(election::Election, config::Abstract_model_config)

Initializes the model with the given election and configuration.

# Arguments
- `election::Election`: The election to initialize the model with.
- `config::Abstract_model_config`: The configuration to initialize the model with.
"""
function init_model(election::Election, config::Abstract_model_config)
	throw(NotImplementedError("init_model"))
end

"""
    select_neighbor(self, model::Abstract_model; rng=Random.GLOBAL_RNG)

Selects a neighbor of the given voter in the given model.

# Arguments
- `self`: The voter to select a neighbor of.
- `model::Abstract_model`: The model to select a neighbor in.
- `rng=Random.GLOBAL_RNG`: The random number generator to use.

# Returns
- `neighbor::Abstract_voter`: The selected neighbor.
"""
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
