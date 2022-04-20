abstract type Abstract_model end

voters(model::T) where T <: Abstract_model = model.voters 
social_network(model::T) where T <: Abstract_model = model.social_network

function graph_diffusion!(model::T, graph_diff_config::U) where {T <: Abstract_model, U <: Abstract_graph_diff_config}
    throw(NotImplementedError("graph_diffusion!"))
end

function get_votes(voters::Vector{T}) where T <: Abstract_voter
    votes = Vector{Vector{Vector{Int64}}}(undef, length(voters))
    for (i, voter) in enumerate(voters)
        votes[i] = get_vote(voter)
    end
    
    return votes
end

function run!(model::T, diffusion_config, logger=nothing::Union{Nothing, Logger}) where T<:Abstract_model
    diffusion!(model, diffusion_config)

    if logger !== nothing && logger.diff_counter[1] % diffusion_config.checkpoint == 0
        logger.diff_counter[1] += 1
        save_log(logger, model)
    end

    return model, logger
end

function ensemble_model(model_dir, ensemble_size)
	models = Vector{Abstract_model}(undef, ensemble_size)
	loggers = Vector{Logger}(undef, ensemble_size)
	
	for i in 1:ensemble_size
		models[i], loggers[i] = restart_model(model_dir)
	end
	
	models, loggers
end

function ensemble_model(model::Abstract_model, ensemble_size)
	models = Vector{Abstract_model}(undef, ensemble_size)
	
	for i in 1:ensemble_size
		models[i] = deepcopy(model)
	end
	
	models
end

function run_ensemble!(models::Vector{T}, diffusion_config, loggers=nothing::Union{Nothing, Vector{Logger}}) where T<:Abstract_model
    for (model, logger) in zip(models, loggers)
        run!(model, diffusion_config, logger)
    end

    return models, loggers
end

function diffusion!(model::T, diffusion_config) where T <: Abstract_model
    voter_diffusion!(model, diffusion_config.voter_diff_config)
    graph_diffusion!(model, diffusion_config.graph_diff_config)
end

function voter_diffusion!(model::T, voter_diff_config::U) where 
    {T <: Abstract_model, U <: Abstract_voter_diff_config}
    sample_size = ceil(Int, voter_diff_config.evolve_vertices * length(model.voters))
    vertex_ids = StatsBase.sample(1:length(model.voters), sample_size, replace=true)

    for id in vertex_ids
        step!(model.voters[id], model.voters, model.social_network, voter_diff_config)
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