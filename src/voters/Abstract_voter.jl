abstract type Abstract_voter end

function get_opinion(voter::Abstract_voter)
    return voter.opinion
end

function get_opinion(voters::Vector{T}) where T <: Abstract_voter
    return [get_opinion(voter) for voter in voters]
end

function get_distance(voter_1::Abstract_voter, voter_2::Abstract_voter)
    return get_distance(get_opinion(voter_1), get_opinion(voter_2))
end

function get_distance(opinion_1::Vector{T}, opinion_2::Vector{T}) where T <: Real
    return Distances.evaluate(Distances.Cityblock(), opinion_1, opinion_2)
end

function get_distance(voter::T, voters::Vector{T}) where T <: Abstract_voter
    return Distances.colwise(Distances.Cityblock(), get_opinion(voter), reduce(hcat, get_opinion(voters)))
end

function get_distance(voters::Vector{T}) where T <: Abstract_voter
    return Distances.pairwise(Distances.Cityblock(), reduce(hcat, get_opinion(voters)), dims=2)
end

function init_voters(election, can_count, voter_config::T) where T <: Abstract_voter_init_config
    throw(NotImplementedError("init_voters"))
end

function step!(self::T, model, voter_diff_config::U) where 
    {T <: Abstract_voter, U <: Abstract_voter_diff_config}
    throw(NotImplementedError("step!(self::T, model, voter_diff_config::U) where {T <: Abstract_voter, U <: Abstract_voter_diff_config}"))
end

function get_vote(voter::Abstract_voter; kwargs...) :: Vote
    throw(NotImplementedError("get_vote"))
end

function get_votes(voters::Vector{T}; kwargs...) where T <: Abstract_voter
    votes = Vector{Vote}(undef, length(voters))
    for (i, voter) in enumerate(voters)
        votes[i] = get_vote(voter; kwargs...)
    end
    
    return votes
end