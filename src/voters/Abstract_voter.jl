abstract type Abstract_voter end
abstract type Abstract_voter_init_config end
abstract type Abstract_voter_diff_config end

function init_voters(election, can_count, voter_config::T) where T <: Abstract_voter_init_config
    throw(NotImplementedError("init_voters"))
end

function step!(self::T, model, voter_diff_config::U) where 
    {T <: Abstract_voter, U <: Abstract_voter_diff_config}
    throw(NotImplementedError("step!(self::T, model, voter_diff_config::U) where {T <: Abstract_voter, U <: Abstract_voter_diff_config}"))
end

function get_ID(voter::T) where T <: Abstract_voter
    return voter.ID
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

function get_opinion(voter::Abstract_voter)
    return voter.opinion
end

function get_opinion(voters::Vector{T}) where T <: Abstract_voter
    return [get_opinion(voter) for voter in voters]
end

get_distance(voter_1::T, voter_2::T) where T <: Abstract_voter = get_distance(get_opinion(voter_1), get_opinion(voter_2))
function get_distance(opinion_1::Vector{T}, opinion_2::Vector{T}) where T <: Real
    return Distances.evaluate(Distances.Cityblock(), opinion_1, opinion_2)
end

get_distance(voter::T, voters::Vector{T}) where T <: Abstract_voter = get_distance(get_opinion(voter), get_opinion(voters))
function get_distance(opinion::Vector{T}, opinions::Vector{Vector{T}}) where T <: Real
    return Distances.colwise(Distances.Cityblock(), opinion, reduce(hcat, opinions))
end

get_distance(voters::Vector{T}) where T <: Abstract_voter = get_distance(get_opinion(voters))
function get_distance(opinions::Vector{Vector{T}}) where T <: Real
    return Distances.pairwise(Distances.Cityblock(), reduce(hcat, opinions), dims=2)
end

get_distance(voters_1::Vector{T}, voters_2::Vector{T}) where T <: Abstract_voter = get_distance(get_opinion(voters_1), get_opinion(voters_2))
function get_distance(opinions_1::Vector{Vector{T}}, opinions_2::Vector{Vector{T}}) where T <: Real
    return Distances.pairwise(Distances.Cityblock(), reduce(hcat, opinions_1), reduce(hcat, opinions_2), dims=2)
end

get_avg_distance(voters_1::Vector{T}, voters_2::Vector{T}) where T <: Abstract_voter = get_avg_distance(get_opinion(voters_1), get_opinion(voters_2))
function get_avg_distance(opinions_1::Vector{Vector{T}}, opinions_2::Vector{Vector{T}}) where T <: Real
    distance_matrix = get_distance(opinions_1, opinions_2)
    return sum(distance_matrix) / length(distance_matrix)
end

get_avg_distance(voters::Vector{T}) where T <: Abstract_voter = get_avg_distance(get_opinion(voters))
function get_avg_distance(opinions::Vector{T}) where T <: Real
    distance_matrix = get_distance(opinions)
    n = size(distance_matrix, 1)
    return sum(distance_matrix) / (n * (n - 1))
end

function get_median_distance(distance_matrix)
    n = size(distance_matrix, 1)
    return Statistics.median(sort(vec(distance_matrix))[n + 1:end])
end