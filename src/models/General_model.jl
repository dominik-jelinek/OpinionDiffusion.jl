struct General_model <: Abstract_model
    voters::Vector{Abstract_voter}
    social_network::MetaGraphs.MetaGraph
    can_count::Int64
end

@kwdef struct General_model_config
    voter_init_config::Abstract_voter_init_config
    graph_init_config::Abstract_graph_init_config
end

@kwdef struct General_graph_diff_config <: Abstract_graph_diff_config
    popularity_ratio::Float64
end

function General_model(election, can_count::Int64, model_config)
    println("Initializing voters:")
    @time voters = init_voters(election, can_count, model_config.voter_init_config)
    
    println("Initializing graph:")
    @time social_network = init_graph(voters, model_config.graph_init_config)

    return General_model(voters, social_network, can_count)
end

function General_model(election, voter_init_config, social_network, can_count::Int64)
    if length(election) != ne(social_network)
        error("Number of voters does not equal the number of vertices")
        return
    end

    println("Initializing voters:")
    @time voters = init_voters(election, can_count, voter_init_config)
    
    return General_model(voters, social_network, can_count)
end

function General_model(voters, graph_init_config::Abstract_graph_init_config , can_count::Int64)
    println("Initializing graph:")
    @time social_network = generate_graph(voters, graph_init_config)

    return General_model(voters, social_network, can_count)
end

"""
Pick a random voter remove one edge based on inverse that it was created and the add one edge
"""
function graph_diffusion!(model::General_model, graph_diff_config::General_graph_diff_config)
    sample_size = ceil(Int, graph_diff_config.evolve_edges * length(model.voters))
    vertex_ids = StatsBase.sample(1:length(model.voters), sample_size, replace=true)

    for id in vertex_ids
        edge_diffusion!(model.voters[id], model.voters, model.social_network, graph_diff_config.popularity_ratio)
    end
end

function edge_diffusion!(self, voters, social_network, popularity_ratio)
    ID = self.ID
    distances = get_distance(self, voters)

    #remove one neighboring edge
    neibrs = neighbors(social_network, self.ID)
    
    degree_probs = 1 ./ degree(social_network, neibrs)
    distance_probs = 2 .^ distances[neibrs]
    probs = popularity_ratio .* degree_probs ./ sum(degree_probs) + (1.0 - popularity_ratio) .* distance_probs ./ sum(distance_probs)
    probs[ID] = 0.0

    to_remove = rand(Distributions.Categorical(probs))
    rem_edge!(social_network, self.ID, neibrs[to_remove])
    
    #add edge
    degree_probs = degree(social_network)
    distance_probs = (1/2) .^ distances
    probs = popularity_ratio .* degree_probs ./ sum(degree_probs) + (1.0 - popularity_ratio) .* distance_probs ./ sum(distance_probs)
    probs[ID] = 0.0
    probs[neighbors(social_network, self.ID)] .= 0.0
    
    add_edge!(social_network, self.ID, rand(Distributions.Categorical(probs)))
end
