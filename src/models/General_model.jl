struct General_model <: Abstract_model
    voters::Vector{Abstract_voter}
    social_network::AbstractGraph
    candidates::Vector{Candidate}
end

@kwdef struct General_model_config <: Abstract_model_config
    voter_init_config::Abstract_voter_init_config
    graph_init_config::Abstract_graph_init_config
end

function init_model(election, candidates, model_config::General_model_config)
    #println("Initializing voters:")
    voters = init_voters(election, model_config.voter_init_config)

    #println("Initializing graph:")
    social_network = init_graph(voters, model_config.graph_init_config)

    return General_model(voters, social_network, candidates)
end
