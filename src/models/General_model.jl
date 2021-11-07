struct General_model <: Abstract_model
    voters::Vector{Abstract_voter}
    social_network::LightGraphs.SimpleGraph
end

function General_model(election, can_count::Int64, model_config)
    println("Initializing voters:")
    @time voters = init_voters(election, can_count, model_config.voter_config)
    
    println("Initializing graph:")
    @time social_network = init_graph(voters, model_config.m)

    model = General_model(voters, social_network)
    return model
end

function graph_diffusion!(model::General_model, edge_diff_config)
    edge_diff_func = edge_diff_config.edge_diff_func
    dist_metric = edge_diff_config.dist_metric
    
    n = edge_diff_config.evolve_edges
    start = rand(1:length(model.voters), n)
    finish = rand(1:length(model.voters), n)

    for i in 1:n
        edge_diffusion!(model.voters[start[i]], model.voters[finish[i]], model.social_network, edge_diff_func, dist_metric)
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
            LightGraphs.rem_edge!(g, voter_1.ID, voter_2.ID)
        end
    else
        if edge_diff_func(distance) * openmindedness > rand()
            LightGraphs.add_edge!(g, voter_1.ID, voter_2.ID)
        end
    end
end