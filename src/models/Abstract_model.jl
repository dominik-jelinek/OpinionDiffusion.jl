get_voters(model::T) where {T<:Abstract_model} = model.voters
get_social_network(model::T) where {T<:Abstract_model} = model.social_network
get_candidates(model::T) where {T<:Abstract_model} = model.candidates

#=
function run_ensemble(
    ensemble_size,
    ensemble_mode,
    diffusions,
    election,
    candidates,
    init_metrics,
    update_metrics!,
    model_configs::Vector{Abstract_model_config},
    init_diff_configs::Vector{Abstract_diff_init_config},
    diff_configs::Vector{Abstract_diff_config},
    log=false
)

    models = Vector{Any}(undef, ensemble_size)
    metrics = Vector{Any}(undef, ensemble_size)
    if length(model_configs) == 1
        models[1] = init_model(election, candidates, model_config)
        metrics[1] = init_metrics(models[1])
    else
        @threads for (i, model_config) in enumerate(model_configs)
            models[i] = init_model(election, candidates, model_config)
            metrics[i] = init_metrics(models[i])
        end
    end

    if length(init_diff_configs) == 1
        init_diffusion!(models[1], init_diff_configs[1])

        if ensemble_mode == "model"
            @threads for i in 2:ensemble_size
                init_diffusion!(models[i], init_diff_configs[1])
            end
        end
    else
        @threads for (i, init_diff_config) in enumerate(init_diff_configs)
            if i != 1
                models[i] = deepcopy(models[1])
                metrics[i] = deepcopy(metrics[1])
            end
            init_diffusion!(models[i], init_diff_config)
        end
    end


    if length(diff_configs) == 1
        # deepcopy diffusion config for the same rng
        actions = run!(models[1], deepcopy(diff_configs[1]), diffusions; metrics=metrics[1], (update_metrics!)=update_metrics!)

        if ensemble_mode == "model" || ensemble_mode == "init_diffusion"
            @threads for i in 2:ensemble_size
                actions = run!(models[i], deepcopy(diff_configs[1]), diffusions; metrics=metrics[i], (update_metrics!)=update_metrics!)
            end
        end
    else
        for i in 2:ensemble_size
            models[i] = deepcopy(models[1])
            metrics[i] = deepcopy(metrics[1])
        end

        @threads for (i, diff_config) in enumerate(diff_configs)
            actions = run!(models[i], diff_config, diffusions; metrics=metrics[i], (update_metrics!)=update_metrics!)
        end
    end

    if log
        save_ensemble(election, model_configs, init_diff_configs, diff_configs, metrics)
    end

    return metrics
end

function run_ensemble(
    model,
    ensemble_size,
    diffusions,
    init_metrics,
    update_metrics!,
    init_diff_configs::Vector{Abstract_diff_init_config},
    diff_configs::Vector{Abstract_diff_config},
    log=false
)

    if length(init_diff_configs) == ensemble_size
        mode = "init_diffusion"
        if length(model_configs) != 1 || length(diff_configs) != 1
            throw(ArgumentError("model_configs and diff_configs must be of length 1"))
        end
    elseif length(diff_configs) == ensemble_size
        mode = "diffusion"
        if length(model_configs) != 1 || length(init_diff_configs) != 1
            throw(ArgumentError("model_configs and init_diff_configs must be of length 1"))
        end
    else
        throw(ArgumentError("ensemble_size must be equal to the length of one of the following: model_configs, init_diff_configs, diff_configs"))
        return
    end

    models = Vector{Any}(undef, ensemble_size)
    metrics = Vector{Any}(undef, ensemble_size)
    models[1] = model
    metrics[1] = init_metrics(model)

    if length(init_diff_configs) == 1
        init_diffusion!(models[1], init_diff_configs[1])
    else
        @threads for (i, init_diff_config) in enumerate(init_diff_configs)
            if i != 1
                models[i] = deepcopy(models[1])
                metrics[i] = deepcopy(metrics[1])
            end

            init_diffusion!(models[i], init_diff_config)
        end
    end


    if length(diff_configs) == 1
        # deepcopy diffusion config for the same rng
        actions = run!(models[1], deepcopy(diff_configs[1]), diffusions; metrics=metrics[1], (update_metrics!)=update_metrics!)

        if mode == "init_diffusion"
            @threads for i in 2:ensemble_size
                actions = run!(models[i], deepcopy(diff_configs[1]), diffusions; metrics=metrics[i], (update_metrics!)=update_metrics!)
            end
        end
    else
        for i in 2:ensemble_size
            models[i] = deepcopy(models[1])
            metrics[i] = deepcopy(metrics[1])
        end

        @threads for (i, diff_config) in enumerate(diff_configs)
            actions = run!(models[i], diff_config, diffusions; metrics=metrics[i], (update_metrics!)=update_metrics!)
        end
    end

    if log
        save_ensemble(election, model_configs, init_diff_configs, diff_configs, metrics)
    end

    return metrics
end

function run_ensemble_model(
    ensemble_size,
    diffusions,
    election,
    candidates,
    init_metrics,
    update_metrics!,
    model_config::Abstract_model_config,
    init_diff_config,
    diff_configs,
    log=false
)

    ens_metrics = Vector{Any}(undef, ensemble_size)

    @threads for i in 1:ensemble_size
        model_seed = rand(UInt32)
        model_rng = MersenneTwister(model_seed)
        model = init_model(election, candidates, model_config; rng=model_rng)
        metrics = init_metrics(model)

        init_diffusion!(model, init_diff_config; rng=model_rng)

        diffusion_seed = rand(UInt32)
        rng = MersenneTwister(diffusion_seed)

        actions = run!(model, diff_configs, diffusions; metrics=metrics, (update_metrics!)=update_metrics!, rng=rng)

        frequent_votes = get_frequent_votes(get_votes(get_voters(model)), 10)
        ens_metrics[i] = Dict("model_seed" => model_seed,
            "diffusion_seed" => diffusion_seed,
            "metrics" => metrics,
            "frequent_votes" => frequent_votes)
    end

    if log
        save_ensemble(model_config, init_diff_configs, diff_configs, ens_metrics)
    end

    return ens_metrics
end

function run_ensemble(model::Abstract_model, ensemble_size, diffusions, init_metrics, update_metrics!, diff_configs, logger=nothing)
    ens_metrics = Vector{Any}(undef, ensemble_size)

    init_diffusion!(model, init_diff_configs; rng=model_rng)

    @threads for i in 1:ensemble_size
        model_cp = deepcopy(model)
        metrics = deepcopy(init_metrics)

        diffusion_seed = rand(UInt32)
        rng = MersenneTwister(diffusion_seed)

        actions = run!(model_cp, diff_configs, diffusions; metrics=metrics, (update_metrics!)=update_metrics!, rng=rng)

        frequent_votes = get_frequent_votes(get_votes(get_voters(model_cp)), 10)
        ens_metrics[i] = Dict("diffusion_seed" => diffusion_seed,
            "metrics" => metrics,
            "frequent_votes" => frequent_votes)
    end

    if logger !== nothing
        save_ensemble(logger.model_dir, diff_configs, ens_metrics)
    end

    return ens_metrics
end
=#
function run!(model::T, diff_configs, diffusions; logger=nothing, checkpoint=1, accumulator=nothing, get_metrics=nothing) where {T<:Abstract_model}
    actions = Vector{Vector{Action}}()
    
    for j in 1:diffusions
        push!(actions, _run!(model, diff_configs; logger=logger, checkpoint=checkpoint, accumulator=accumulator, get_metrics=get_metrics))
    end

    return actions
end

function _run!(model::T, diff_configs::Vector{Abstract_diff_config}; logger=nothing, checkpoint=1, accumulator=nothing, get_metrics=nothing) where {T<:Abstract_model}
    actions = diffusion!(model, diff_configs)

    if get_metrics !== nothing
        add_metrics!(accumulator, get_metrics(model))
    end

    if logger !== nothing
        logger.diff_counter[1] += 1

        if logger.diff_counter[1] % checkpoint == 0
            save_log(logger, model)
        end
    end

    return actions
end

function save_log(model::T, model_dir) where {T<:Abstract_model}
    jldsave("$(model_dir)/model_init.jld2"; model)
end

function save_log(model::T, exp_dir::String, idx::Int64) where {T<:Abstract_model}
    jldsave("$(exp_dir)/model_$(idx).jld2"; model)
end

#restart
function load_log(model_dir::String)
    return load("$(model_dir)/model_init.jld2", "model")
end

#load
function load_log(exp_dir::String, idx::Int64)
    if idx == -1
        idx = last_log_idx(exp_dir)
    end

    return load("$(exp_dir)/model_$(idx).jld2", "model")
end

function load_log(exp_dir::String, model_name::String)
    return load("$(exp_dir)/$(model_name)", "model")
end

function load_logs(exp_dir::String, start_idx::Int64, end_idx::Int64)
    if end_idx == -1
        end_idx = last_log_idx(exp_dir)
    end
    models = Vector{Abstract_model}(undef, end_idx - start_idx + 1)

    for (i, j) in enumerate(start_idx:end_idx)
        models[i] = load_log(exp_dir, j)
    end

    return models
end

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

function pipeline(experiment_config)
	election = pipeline_election(input_filename, selection_configs)
	model, logger = pipeline_model(election, voter_init_configs, graph_init_configs)
	accumulator = pipeline_diffusion(model, init_diff_configs, diff_configs)
	gathered_metrics = gather_metrics(accumulator)

	return gathered_metrics
end

function pipeline_election(input_filename, selection_configs)
	election = parse_data("./data/" * input_filename)

	elections = []
    selection_configs = []

	for selection_config in selection_configs
		push!(election, select(election, selection_config))
        push!(selection_configs, selection_config)
	end
	
	return elections, selection_configs
end

function pipeline_model(elections, voter_init_configs, graph_init_configs)
	pairs = []
	for voter_init_config in voter_init_configs
		for election in elections
			push!(models, (election, init_voters(election.votes, voter_init_config)))
		end
	end

	triplets = []
	for graph_init_config in graph_init_configs
		for (election, voters) in pairs
			push!(triplets, (election, voters, init_graph(voters, graph_init_config)))
		end
	end

	models = []
	for (election, voters, social_network) in models
		push!(models, General_model(voters, social_network, election.party_names, election.candidates))
	end

	return models
end

function pipeline_diffusion(models, diff_init_configs, diff_configs, get_metrics)
    initialized_models = []
    accumulators = []
    for model in models
        for diff_init_config in diff_init_configs
            copied = deepcopy(model)
            init_diffusion!(copied, diff_init_config)
            push!(initialized_models, copied)

            accumulator = init_accumulator(get_metrics(copied))
            push!(accumulators, accumulator)
        end
    end
	
	for (model, accumulator) in zip(initialized_models, accumulators)
        for diff_config in diff_configs
            run!(model, diff_config, diffusions; accumulator=accumulator, get_metrics=get_metrics)
        end
    end

	return accumulators
end