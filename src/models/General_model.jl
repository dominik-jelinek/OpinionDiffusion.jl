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