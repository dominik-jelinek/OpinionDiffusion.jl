@kwdef struct Action
    operation::String
    ID::Union{Int64, Tuple{Int64, Int64}}
    old
    new
end

get_voters(model::T) where T <: Abstract_model = model.voters 
get_social_network(model::T) where T <: Abstract_model = model.social_network
get_candidates(model::T) where T <: Abstract_model = model.candidates

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

function init_model(election, candidates, model_config; rng=Random.GLOBAL_RNG)
    throw(NotImplementedError("init_model"))
end
init_model(election, candidates, model_config, seed) = init_model(election, candidates, model_config; rng=Random.MersenneTwister(seed))

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
    
        actions = run!(model, diff_configs, diffusions; metrics=metrics, update_metrics! =update_metrics!, rng=rng)

        frequent_votes = get_frequent_votes(get_votes(get_voters(model_cp)), 10)
        ens_metrics[i] = Dict(  "model_seed" => model_seed, 
                                "diffusion_seed" => diffusion_seed, 
                                "metrics" => metrics,
                                "frequent_votes" => frequent_votes)
    end

	if log
		save_ensemble(model_config, diff_configs, ens_metrics)
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

        actions = run!(model_cp, diff_configs, diffusions; metrics=metrics, update_metrics! =update_metrics!, rng=rng)

        frequent_votes = get_frequent_votes(get_votes(get_voters(model_cp)), 10)
        ens_metrics[i] = Dict(  "diffusion_seed" => diffusion_seed, 
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

function run!(model::T, diff_configs, diffusions; logger=nothing, checkpoint=1, metrics=nothing, update_metrics! =nothing, rng=Random.GLOBAL_RNG) where T<:Abstract_model
    actions = Vector{Vector{Action}}()

    for j in 1:diffusions
        push!(actions, _run!(model, diff_configs; logger=logger, checkpoint=checkpoint, metrics=metrics, update_metrics! = update_metrics!, rng=rng))
    end

    return actions
end

function _run!(model::T, diff_configs::Vector{Abstract_diff_config}; logger=nothing, checkpoint=1, metrics=nothing, update_metrics! =nothing, rng=Random.GLOBAL_RNG) where T<:Abstract_model
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

function save_log(model::T, model_dir) where T<:Abstract_model
    jldsave("$(model_dir)/model_init.jld2"; model)
end

function save_log(model::T, exp_dir::String, idx::Int64) where T<:Abstract_model
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