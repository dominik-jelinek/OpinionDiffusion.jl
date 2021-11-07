abstract type Abstract_model end

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

function get_opinions(voters::Vector{T}) where T <: Abstract_voter
    return reduce(hcat, [voter.opinion for voter in voters])
end

function run!(model::T, diffusion_config, logger=nothing::Union{Nothing, Logger}) where T<:Abstract_model
    diffusion!(model, diffusion_config)

    if logger !== nothing && logger.diff_counter[1] % diffusion_config.diffusions == 0
        save_log(logger, model)
    end
end

function diffusion!(model::T, diffusion_config) where T <: Abstract_model
    voter_diffusion!(model, diffusion_config.voter_diff_config)
    graph_diffusion!(model, diffusion_config.graph_diff_config)
end

function voter_diffusion!(model::T, voter_diff_config::U) where 
    {T <: Abstract_model, U <: Abstract_voter_diff_config}
    vertexes = rand(1:length(model.voters), voter_diff_config.evolve_vertices)

    for v in vertexes
        step!(model.voters[v], model.voters, model.social_network, voter_diff_config)
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