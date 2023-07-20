struct General_model <: Abstract_model
	voters::Vector{Abstract_voter}

	social_network::AbstractGraph

	candidates::Vector{Candidate}
end

@kwdef struct General_model_config <: Abstract_model_config
	voter_config::Abstract_voter_config
	graph_config::Abstract_graph_config
end

"""
    init_model(election::Election, config::General_model_config)

Initializes the model with the given election and configuration.

# Arguments
- `election::Election`: The election to initialize the model with.
- `config::General_model_config`: The configuration to initialize the model with.

# Returns
- `model::General_model`: The initialized model.
"""
function init_model(election::Election, config::General_model_config)
	# voters
	votes = get_votes(election)
	voters = init_voters(votes, config.voter_config)

	# graph
	social_network = init_graph(voters, config.graph_config)

	return General_model(voters, social_network, get_candidates(election))
end
