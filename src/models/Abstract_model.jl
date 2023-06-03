get_voters(model::T) where {T<:Abstract_model} = model.voters
get_social_network(model::T) where {T<:Abstract_model} = model.social_network
get_candidates(model::T) where {T<:Abstract_model} = model.candidates

function init_model(election, candidates, model_config; rng=Random.GLOBAL_RNG)
    throw(NotImplementedError("init_model"))
end
init_model(election, candidates, model_config, seed) = init_model(election, candidates, model_config; rng=Random.MersenneTwister(seed))

function run_ensemble(
    ensemble_size,
    diffusions,
    election,
    candidates,
    init_metrics,
    update_metrics!,
    model_configs::Vector{Abstract_model_config},
    init_diff_configs::Vector{Abstract_init_diff_config},
    diff_configs::Vector{Abstract_diff_config},
    log=false)

    if length(model_configs) == ensemble_size
        mode = "model"
        if length(init_diff_configs) != 1 || length(diff_configs) != 1
            throw(ArgumentError("init_diff_configs and diff_configs must be of length 1"))
        end
    elseif length(init_diff_configs) == ensemble_size
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
    if length(model_configs) == 1
        model_seed = rand(UInt32)
        model_rng = MersenneTwister(model_seed)
        models[1] = init_model(election, candidates, model_config; rng=model_rng)
    else
        @threads for (i, model_config) in enumerate(model_configs)
            model_seed = rand(UInt32)
            model_rng = MersenneTwister(model_seed)
            models[i] = init_model(election, candidates, model_config; rng=model_rng)
            metrics[i] = init_metrics(model)
        end
    end

    if length(init_diff_configs) == 1
        init_diffusion!(models[1], init_diff_configs[1])

        if mode == "model"
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
        diffusion_seed = rand(UInt32)
        rng = MersenneTwister(diffusion_seed)
        actions = run!(models[1], diff_configs[1], diffusions; metrics=metrics[1], (update_metrics!)=update_metrics!, rng=rng)

        if mode == "model" || mode == "init_diffusion"
            @threads for i in 2:ensemble_size
                diffusion_seed = rand(UInt32)
                rng = MersenneTwister(diffusion_seed)
                actions = run!(models[i], diff_configs[1], diffusions; metrics=metrics[i], (update_metrics!)=update_metrics!, rng=rng)
            end
        end
    else
        @threads for (i, diff_config) in enumerate(diff_configs)
            if i != 1
                models[i] = deepcopy(models[1])
                metrics[i] = deepcopy(metrics[1])
            end
            diffusion_seed = rand(UInt32)
            rng = MersenneTwister(diffusion_seed)
            actions = run!(models[i], diff_config, diffusions; metrics=metrics[i], (update_metrics!)=update_metrics!, rng=rng)
        end
    end

    if log
        save_ensemble(model_configs, init_diff_configs, diff_configs, metrics)
    end

    return metrics
end

function run_ensemble_model(ensemble_size, diffusions, election, candidates, init_metrics, update_metrics!, model_config, init_diff_configs, diff_configs, log=false)
    ens_metrics = Vector{Any}(undef, ensemble_size)

    @threads for i in 1:ensemble_size
        model_seed = rand(UInt32)
        model_rng = MersenneTwister(model_seed)
        model = init_model(election, candidates, model_config; rng=model_rng)
        metrics = init_metrics(model)

        init_diffusion!(model, init_diff_configs; rng=model_rng)

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

function run(election, candidates, model_config, model_seed, init_diff_configs, diff_configs, diffusion_seed, diffusions)
    model_rng = MersenneTwister(model_seed)
    model = init_model(election, candidates, model_config; rng=model_rng)
    logger = Logger(model)

    init_diffusion!(model, init_diff_configs; rng=model_rng)
    diff_rng = MersenneTwister(diffusion_seed)
    actions = run!(model, diff_configs, diffusions; logger=logger, rng=diff_rng)

    return logger, actions
end

function run!(model::T, diff_configs, diffusions; logger=nothing, checkpoint=1, metrics=nothing, (update_metrics!)=nothing, rng=Random.GLOBAL_RNG) where {T<:Abstract_model}
    actions = Vector{Vector{Action}}()

    for j in 1:diffusions
        push!(actions, _run!(model, diff_configs; logger=logger, checkpoint=checkpoint, metrics=metrics, (update_metrics!)=update_metrics!, rng=rng))
    end

    return actions
end

function _run!(model::T, diff_configs::Vector{Abstract_diff_config}; logger=nothing, checkpoint=1, metrics=nothing, (update_metrics!)=nothing, rng=Random.GLOBAL_RNG) where {T<:Abstract_model}
    actions = diffusion!(model, diff_configs; rng=rng)

    if metrics !== nothing
        update_metrics!(model, metrics)
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