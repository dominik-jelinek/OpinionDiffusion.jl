struct General_model <: Abstract_model
    voters::Vector{Abstract_voter}
    social_network::SimpleWeightedGraphs.SimpleWeightedGraph
end

function General_model(election, can_count::Int64, model_config)
    println("Initializing voters:")
    @time voters = init_voters(election, can_count, model_config.voter_config)
    
    println("Initializing graph:")
    @time social_network = weighted_barabasi_albert_graph(voters, model_config.m)

    return General_model(voters, social_network)
end

function graph_diffusion!(model::General_model, graph_diff_config::General_graph_diff_config)
    edge_diff_func = graph_diff_config.edge_diff_func
    dist_metric = graph_diff_config.dist_metric
    
    sample_size = ceil(Int, graph_diff_config.evolve_edges * length(model.voters))
    start_ids = StatsBase.sample(1:length(model.voters), sample_size, replace=true)
    finish_ids = StatsBase.sample(1:length(model.voters), sample_size, replace=true)

    for i in 1:sample_size
        edge_diffusion!(model.voters[start_ids[i]], model.voters[finish_ids[i]], model.social_network, edge_diff_func, dist_metric)
    end
end

function edge_diffusion!(voter_1, voter_2, g, edge_diff_func, dist_metric::Distances.Metric)
    if voter_1.ID == voter_2.ID
        return
    end
    distance = Distances.evaluate(dist_metric, voter_1.opinion, voter_2.opinion)
    openmindedness = voter_1.openmindedness + voter_2.openmindedness

    if has_edge(g, voter_1.ID, voter_2.ID)
        if edge_diff_func(distance) * openmindedness < rand()
            Graphs.rem_edge!(g, voter_1.ID, voter_2.ID)
        end
    else
        if edge_diff_func(distance) * openmindedness > rand()
            Graphs.add_edge!(g, voter_1.ID, voter_2.ID)
        end
    end
end

"""
Pick a random voter remove one edge based on inverse that it was created and the add one edge
"""
function graph_diffusion!(model::General_model, graph_diff_config::General_graph_diff_config)
    edge_diff_func = graph_diff_config.edge_diff_func
    dist_metric = graph_diff_config.dist_metric
    
    sample_size = ceil(Int, graph_diff_config.evolve_edges * length(model.voters))
    vertex_ids = StatsBase.sample(1:length(model.voters), sample_size, replace=true)

    for id in vertex_ids
        edge_diffusion!(model.voters[id], model.voters, model.social_network)
    end
end

function edge_diffusion!(self, voters, social_network)
    #remove edge
    neibrs = neighbors(social_network, self.ID)
    
    probs = Vector{Float64}(undef, length(neibrs))
    for i in 1:length(neibrs)
        probs[i] = degree(social_network, neibrs[i]) * (1.0 + get_distance(self, voters[neibrs[i]]))
    end
    
    probs = probs ./ sum(probs)
    to_remove = rand(Distributions.Categorical(probs))
    rem_edge!(social_network, (self.ID, neibrs[to_remove]))
    
    #add edge
    probs = Vector{Float64}(undef, length(voters))
    for i in 1:length(voters)
        probs[i] = degree(social_network, voters[i]) / (1.0 + get_distance(self, voters[i]))
    end
    probs[self.ID] = 0.0
    probs = probs ./ sum(probs)
    add_edge!(social_network, (self.ID, rand(Distributions.Categorical(probs))))
end