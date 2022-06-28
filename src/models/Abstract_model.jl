abstract type Abstract_model end

get_voters(model::T) where T <: Abstract_model = model.voters 
get_social_network(model::T) where T <: Abstract_model = model.social_network

function graph_diffusion!(model::T, evolve_edges, graph_diff_config::U) where {T <: Abstract_model, U <: Abstract_graph_diff_config}
    throw(NotImplementedError("graph_diffusion!"))
end

function run!(model::T, diffusion_config, logger=nothing::Union{Nothing, Logger}) where T<:Abstract_model
    diffusion!(model, diffusion_config)

    if logger !== nothing && logger.diff_counter[1] % diffusion_config.checkpoint == 0
        logger.diff_counter[1] += 1
        save_log(logger, model)
    end
end

function run_ensemble!(model::Abstract_model, ensemble_size, diffusions, init_metrics, update_metrics!, diffusion_config)
    metrics_ens = Vector{Any}(undef, ensemble_size)

    @threads for i in 1:ensemble_size
        model_cp = deepcopy(model)
        metrics = deepcopy(init_metrics)

        for j in 1:diffusions
            run!(model_cp, diffusion_config)
            update_metrics!(model_cp, metrics)
        end

        metrics_ens[i] = metrics
    end

    return metrics_ens
end

function diffusion!(model::T, diffusion_config) where T <: Abstract_model
    voter_diffusion!(model, diffusion_config.evolve_vertices, diffusion_config.voter_diff_config)
    graph_diffusion!(model, diffusion_config.evolve_edges, diffusion_config.graph_diff_config)
end

function voter_diffusion!(model::T, evolve_vertices, voter_diff_config::U) where 
    {T <: Abstract_model, U <: Abstract_voter_diff_config}

    voters = get_voters(model)
    social_network = get_social_network(model)

    sample_size = ceil(Int, evolve_vertices * length(get_voters(model)))
    vertex_ids = StatsBase.sample(1:length(get_voters(model)), sample_size, replace=true)
    
    for id in vertex_ids
        step!(voters[id], voters, social_network, model.can_count, voter_diff_config)
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