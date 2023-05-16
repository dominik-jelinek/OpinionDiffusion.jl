struct General_model <: Abstract_model
    voters::Vector{Abstract_voter}
    social_network::AbstractGraph
    candidates::Vector{Candidate}
end

@kwdef struct General_model_config
    voter_init_config::Abstract_voter_init_config
    graph_init_config::Abstract_graph_init_config
end

@kwdef struct General_graph_diff_config <: Abstract_graph_diff_config
    homophily::Float64
end

function init_model(election, candidates, model_config::General_model_config; rng=Random.GLOBAL_RNG)
    #println("Initializing voters:")
    voters = init_voters(election, model_config.voter_init_config; rng=rng)

    #println("Initializing graph:")
    social_network = init_graph(voters, model_config.graph_init_config; rng=rng)

    return General_model(voters, social_network, candidates)
end

"""
    graph_diffusion!(model::General_model, evolve_edges::Float64, graph_diff_config::General_graph_diff_config; rng=Random.GLOBAL_RNG)

Diffuses the graph of the model by modifying edges according to the graph diffusion configuration.

# Arguments
- `model::General_model`: The model to diffuse.
- `evolve_edges::Float64`: The proportion of voters that will have their edges modified.
- `graph_diff_config::General_graph_diff_config`: The configuration of the graph diffusion.
- `rng::AbstractRNG`: The random number generator to use.

# Returns
- ID's of the voters that had their edges modified.
"""
function graph_diffusion!(model::General_model, evolve_edges::Float64, graph_diff_config::General_graph_diff_config; rng=Random.Global)
    actions = Vector{Action}()

    voters = get_voters(model)
    
    sample_size = ceil(Int, evolve_edges * length(voters))
    vertex_ids = StatsBase.sample(rng, 1:length(voters), sample_size, replace=true)

    for id in vertex_ids
        append!(actions, edge_diffusion!(voters[id], model, graph_diff_config.homophily; rng=rng))
    end

    return actions
end

function edge_diffusion!(self, model, popularity_ratio; rng=Random.Global)
    voters, social_network = get_voters(model), get_social_network(model)
    ID = self.ID
    distances = get_distance(self, voters)

    neibrs = neighbors(social_network, ID)
    if length(neibrs) == 0
        # do not add or remove edges because this would change average degree
        return
    end

    #remove one neighboring edge
    degree_probs = 1 ./ degree(social_network, neibrs)
    distance_probs = 2 .^ distances[neibrs]
    probs = popularity_ratio .* degree_probs ./ sum(degree_probs) + (1.0 - popularity_ratio) .* distance_probs ./ sum(distance_probs)

    to_remove = StatsBase.sample(rng, 1:length(neibrs), StatsBase.Weights(probs))
    rem_edge!(social_network, ID, neibrs[to_remove])

    #add edge
    degree_probs = degree(social_network)
    distance_probs = (1 / 2) .^ distances
    probs = popularity_ratio .* degree_probs ./ sum(degree_probs) + (1.0 - popularity_ratio) .* distance_probs ./ sum(distance_probs)
    probs[ID] = 0.0
    probs[neighbors(social_network, ID)] .= 0.0

    to_add = StatsBase.sample(rng, 1:length(voters), StatsBase.Weights(probs))
    add_edge!(social_network, ID, to_add)
end