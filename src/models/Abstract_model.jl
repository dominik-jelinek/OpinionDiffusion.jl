abstract type Abstract_model end

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

function diffusion!(model::T, diffusion_config) where T <: Abstract_model
    voter_diffusion!(model, diffusion_config.voter_diff_config)
    graph_diffusion!(model, diffusion_config.edge_diff_config)
end

function voter_diffusion!(model::T, voter_diff_config) where T <: Abstract_model
    vertexes = rand(1:length(model.voters), voter_diff_config.evolve_vertices)

    for v in vertexes
        step!(model.voters[v], model.voters, model.social_network, voter_diff_config)
    end
end

function graph_diffusion!(model::T, edge_diff_config) where T <: Abstract_model
    throw(NotImplementedError("step!"))
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
    return load("$(exp_dir)/model_$(idx).jld2", "model")
end