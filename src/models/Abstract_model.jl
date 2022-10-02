abstract type Abstract_model end
abstract type Abstract_graph_init_config <: Config end
abstract type Abstract_graph_diff_config <: Config end

@kwdef struct Diffusion_config <: Config
    checkpoint::Int64
    evolve_vertices::Float64
    evolve_edges::Float64
    voter_diff_config::Abstract_voter_diff_config
    graph_diff_config::Abstract_graph_diff_config
end

get_voters(model::T) where T <: Abstract_model = model.voters 
get_social_network(model::T) where T <: Abstract_model = model.social_network

function graph_diffusion!(model::T, graph_diff_config::U) where {T <: Abstract_model, U <: Abstract_graph_diff_config}
    throw(NotImplementedError("graph_diffusion!"))
end

function run!(model::T, diffusion_config; rng=rng) where T<:Abstract_model
    diffusion!(model, diffusion_config; rng=rng)
end

function run!(model::T, diffusion_config, logger; rng=rng) where T<:Abstract_model
    run!(model, diffusion_config; rng=rng)

    logger.diff_counter[1] += 1
        
    if logger.diff_counter[1] % diffusion_config.checkpoint == 0
        save_log(logger, model)
    end
end

function run_ensemble(model::Abstract_model, ensemble_size, diffusions, init_metrics, update_metrics!, diffusion_config, logger=nothing)
    ens_metrics = Vector{Any}(undef, ensemble_size)

    @threads for i in 1:ensemble_size
        model_cp = deepcopy(model)
        metrics = deepcopy(init_metrics)
        
        diffusion_seed = rand(UInt32)
        rng = MersenneTwister(diffusion_seed)

        for j in 1:diffusions
            run!(model_cp, diffusion_config; rng=rng)
            update_metrics!(model_cp, metrics)
        end

        ens_metrics[i] = Dict(  "diffusion_seed" => diffusion_seed, 
                                "metrics" => metrics)
        display(get_frequent_votes(get_votes(get_voters(model_cp)), 5))
        println()
        println(diffusion_seed)
        println("_____________________________________________________")
    end

	if logger != nothing
		save_ensemble(logger.model_dir, diffusion_config, ens_metrics)
	end

    return ens_metrics
end

function run_ensemble_model(ensemble_size, diffusions, election, init_metrics, can_count, update_metrics!, model_config, diffusion_config, log=false)
    ens_metrics = Vector{Any}(undef, ensemble_size)
    
    diffusion_seed = rand(UInt32)

    @threads for i in 1:ensemble_size
        model_seed = rand(UInt32)
        model_rng = MersenneTwister(model_seed)
        model = General_model(election, can_count, model_config; rng=model_rng)
        metrics = init_metrics(model, can_count)

        rng = MersenneTwister(diffusion_seed)
        for j in 1:diffusions
            run!(model, diffusion_config; rng=rng)
            update_metrics!(model, metrics)
        end

        ens_metrics[i] = Dict(  "model_seed" => model_seed, 
                                "diffusion_seed" => diffusion_seed, 
                                "metrics" => metrics)

        display(get_frequent_votes(get_votes(get_voters(model)), 5))
        println()
        println(model_seed)
        println("_____________________________________________________")
    end

	if log
		save_ensemble(model_config, diffusion_config, ens_metrics)
	end

    return ens_metrics
end

function diffusion!(model::T, diffusion_config; rng=Random.GLOBAL_RNG) where T <: Abstract_model
    voter_diffusion!(model, diffusion_config.evolve_vertices, diffusion_config.voter_diff_config; rng=rng)
    graph_diffusion!(model, diffusion_config.evolve_edges, diffusion_config.graph_diff_config; rng=rng)
end

function voter_diffusion!(model::T, evolve_vertices, voter_diff_config::U; rng=Random.Global) where 
    {T <: Abstract_model, U <: Abstract_voter_diff_config}

    voters = get_voters(model)
    sample_size = ceil(Int, evolve_vertices * length(voters))
    vertex_ids = StatsBase.sample(rng, 1:length(voters), sample_size, replace=true)
    
    for id in vertex_ids
        step!(voters[id], model, voter_diff_config; rng=rng)
    end
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