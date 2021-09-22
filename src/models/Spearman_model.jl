struct Spearman_model <: Abstract_model
    voters::Vector{Spearman_voter}
    social_network::SimpleGraph

    log_dir::String
    exp_counter::Vector{Int}
end

function Spearman_model(election, can_count, model_config)
    #init voters
    println("Initializing voters:")
    weight_func = parse_function(model_config["weight_func"])
    
    weights = map(weight_func, 1:can_count)
    openmindedness_distr = Truncated(Normal(0.5, 0.1), 0.0, 1.0)
    stubbornness_distr = Truncated(Normal(0.5, 0.1), 0.0, 1.0)

    @time voters = init_voters(election, weights, openmindedness_distr, stubbornness_distr)

    #init graph
    #println("Initializing edges:")
    #init_edge_func = parse_function(initConfig["init_edge_func"]) 10x slower
    #init_edge_func = x->(1/2)^(x + 5.14)
    #dist_metric = parse_metric(model_config["dist_metric"])
    #edge_limit = 100*length(voters)
    #@time edges = generate_edges(voters, dist_metric, init_edge_func)
    
    println("Initializing graph:")
    
    @time social_network = init_graph(voters, model_config["m"])
    
    #@time social_network = init_graph(length(voters), edges)

    #init logging
    println("Initializing logging")
    
    exp_counter = [1]
    log_dir = "logs/" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(log_dir)

    YAML.write_file("$(log_dir)/model_config.yml", model_config)
    model = Spearman_model(voters, social_network, log_dir, exp_counter)
    @time log(model)
    return model
end

function init_voters(election, weights, openmindedness_distr, stubbornness_distr)
    voters = Vector{Spearman_voter}(undef, length(election))
    for (i, vote) in enumerate(election)
        voters[i] = Spearman_voter(i, vote, weights, openmindedness_distr, stubbornness_distr)
    end

    return voters
end

function graph_diffusion!(model, edge_diff_config)
    edge_diff_func = parse_function(edge_diff_config["edge_diff_func"])
    dist_metric = parse_metric(edge_diff_config["dist_metric"])
    
    n = edge_diff_config["evolve_edges"]
    start = rand(1:length(model.voters), n)
    finish = rand(1:length(model.voters), n)

    for i in 1:n
        edge_diffusion!(model.voters[start[i]], model.voters[finish[i]], model.social_network, edge_diff_func, dist_metric)
    end
end

function edge_diffusion!(voter_1, voter_2, g, edgeDiffFunc, distMetric)
    if voter_1.ID == voter_2.ID
        return
    end
    distance = Distances.evaluate(distMetric, voter_1.opinion, voter_2.opinion)
    openmindedness = voter_1.openmindedness + voter_2.openmindedness

    if has_edge(g, voter_1.ID, voter_2.ID)
        if edgeDiffFunc(distance) * openmindedness < rand()
            rem_edge!(g, voter_1.ID, voter_2.ID)
        end
    else
        if edgeDiffFunc(distance) * openmindedness > rand()
            add_edge!(g, voter_1.ID, voter_2.ID)
        end
    end
end

function log(model::Spearman_model)
    jldsave("$(model.log_dir)/model.jld2"; model)
end