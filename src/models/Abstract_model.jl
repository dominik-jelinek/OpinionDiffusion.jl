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

function save_log(model::T) where T<:Abstract_model
    jldsave("$(model.log_dir)/model_init.jld2"; model)
end

function load_log(model::T) where T<:Abstract_model
    return load("$(model.log_dir)/model_init.jld2", "model")
end