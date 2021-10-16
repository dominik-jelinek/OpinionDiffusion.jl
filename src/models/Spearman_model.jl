struct Spearman_model <: Abstract_model
    voters::Vector{Spearman_voter}
    social_network::LightGraphs.SimpleGraph

    log_dir::String
    exp_counter
end

function Spearman_model(election, can_count::Int64, model_config)
    println("Initializing voters:")
    weights = map(model_config.weight_func, 1:can_count)
    openmindedness_distr = Distributions.Truncated(model_config.openmindedness_distr, 0.0, 1.0)
    stubbornness_distr = Distributions.Truncated(model_config.stubbornness_distr, 0.0, 1.0)

    @time voters = init_voters(election, weights, openmindedness_distr, stubbornness_distr)
    
    println("Initializing graph:")
    @time social_network = init_graph(voters, model_config.m)

    println("Initializing logging")
    log_dir = "logs/" * model_config.log_name * "_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(log_dir)

    YAML.write_file("$(log_dir)/model_config.yml", model_config)
    model = Spearman_model(voters, social_network, log_dir, [0])
    save_log(model)
    return model
end

function init_voters(election, weights, openmindedness_distr, stubbornness_distr)
    voters = Vector{Spearman_voter}(undef, length(election))
    for (i, vote) in enumerate(election)
        voters[i] = Spearman_voter(i, vote, weights, openmindedness_distr, stubbornness_distr)
    end

    return voters
end

function graph_diffusion!(model::Spearman_model, edge_diff_config)
    edge_diff_func = edge_diff_config.edge_diff_func
    dist_metric = edge_diff_config.dist_metric
    
    n = edge_diff_config.evolve_edges
    start = rand(1:length(model.voters), n)
    finish = rand(1:length(model.voters), n)

    for i in 1:n
        edge_diffusion!(model.voters[start[i]], model.voters[finish[i]], model.social_network, edge_diff_func, dist_metric)
    end
end

function edge_diffusion!(voter_1, voter_2, g, edgeDiffFunc, distMetric::Distances.Metric)
    if voter_1.ID == voter_2.ID
        return
    end
    distance = Distances.evaluate(distMetric, voter_1.opinion, voter_2.opinion)
    openmindedness = voter_1.openmindedness + voter_2.openmindedness

    if has_edge(g, voter_1.ID, voter_2.ID)
        if edgeDiffFunc(distance) * openmindedness < rand()
            LightGraphs.rem_edge!(g, voter_1.ID, voter_2.ID)
        end
    else
        if edgeDiffFunc(distance) * openmindedness > rand()
            LightGraphs.add_edge!(g, voter_1.ID, voter_2.ID)
        end
    end
end