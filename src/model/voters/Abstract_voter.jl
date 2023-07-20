"""
	init_voters(votes::Vector{Vote}, voter_config::Abstract_voter_config)

Initializes the voters from the given votes and voter_config.

# Arguments
- `votes::Vector{Vote}`: The votes to initialize the voters with.
- `voter_config::Abstract_voter_config`: The config to initialize the voters with.

# Returns
- `voters::Vector{Abstract_voter}`: The initialized voters.
"""
function init_voters(votes::Vector{Vote}, voter_config::T) where {T<:Abstract_voter_config}
	throw(NotImplementedError("init_voters"))
end

"""
	get_ID(voter::Abstract_voter)::Int

Returns the ID of the given voter.

# Arguments
- `voter::Abstract_voter`: The voter to get the ID of.

# Returns
- `ID::Int`: The ID of the given voter.
"""
function get_ID(voter::T) where {T<:Abstract_voter}
	return voter.ID
end

"""
	get_properties(voter::Abstract_voter)::Dict{Symbol, Any}

Returns the properties of the given voter.

# Arguments
- `voter::Abstract_voter`: The voter to get the properties of.

# Returns
- `properties::Dict{Symbol, Any}`: The properties of the given voter.
"""
function get_properties(voter::T) where {T<:Abstract_voter}
	return voter.properties
end

"""
	get_property(voter::Abstract_voter, property::Symbol)::Any

Returns the given property of the given voter.

# Arguments
- `voter::Abstract_voter`: The voter to get the property of.
- `property::Symbol`: The property to get.

# Returns
- `value::Any`: The value of the given property of the given voter.
"""
function get_property(voter::T, property) where {T<:Abstract_voter}
	return voter.properties[property]
end

"""
	get_property(voters::Vector{Abstract_voter}, property::Symbol)::Vector{Any}

Returns the given property of the given voters.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters to get the property of.
- `property::Symbol`: The property to get.

# Returns
- `values::Vector{Any}`: The values of the given property of the given voters.
"""
function get_property(voters::Vector{T}, property) where {T<:Abstract_voter}
	return [get_property(voter, property) for voter in voters]
end

"""
	set_property!(voter::Abstract_voter, property::Symbol, value::Any)

Sets the given property of the given voter to the given value.

# Arguments
- `voter::Abstract_voter`: The voter to set the property of.
- `property::Symbol`: The property to set.
- `value::Any`: The value to set the property to.
"""
function set_property!(voter::T, property, value) where {T<:Abstract_voter}
	voter.properties[property] = value
end

"""
	set_property!(voters::Vector{Abstract_voter}, property::Symbol, values::Vector{Any})

Sets the given property of the given voters to the given values.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters to set the property of.
- `property::Symbol`: The property to set.
- `values::Vector{Any}`: The values to set the property to.
"""
function set_property!(voters::Vector{T}, property, values::Vector{<:Any}) where {T<:Abstract_voter}
	for (voter, value) in zip(voters, values)
		set_property!(voter, property, value)
	end
end

"""
	get_vote(voter::Abstract_voter)::Vote

Returns the vote of the given voter.

# Arguments
- `voter::Abstract_voter`: The voter to get the vote of.

# Returns
- `vote::Vote`: The vote of the given voter.
"""
function get_vote(voter::Abstract_voter)::Vote
	throw(NotImplementedError("get_vote"))
end

"""
	get_votes(voters::Vector{Abstract_voter})::Vector{Vote}

Returns the votes of the given voters.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters to get the votes of.

# Returns
- `votes::Vector{Vote}`: The votes of the given voters.
"""
function get_votes(voters::Vector{T}) where {T<:Abstract_voter}
	votes = Vector{Vote}(undef, length(voters))
	for (i, voter) in enumerate(voters)
		votes[i] = get_vote(voter)
	end

	return votes
end

"""
	get_opinion(voter::Abstract_voter)::Vector{Real}

Returns the opinion of the given voter.

# Arguments
- `voter::Abstract_voter`: The voter to get the opinion of.

# Returns
- `opinion::Vector{Real}`: The opinion of the given voter.
"""
function get_opinion(voter::Abstract_voter)
	return voter.opinion
end

"""
	get_opinion(voters::Vector{Abstract_voter})::Vector{Vector{Real}}

Returns the opinions of the given voters.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters to get the opinions of.

# Returns
- `opinions::Vector{Vector{Real}}`: The opinions of the given voters.
"""
function get_opinion(voters::Vector{T}) where {T<:Abstract_voter}
	return [get_opinion(voter) for voter in voters]
end

"""
	get_distance(voter_1::Abstract_voter, voter_2::Abstract_voter)::Real

Returns the distance between the given voters.

# Arguments
- `voter_1::Abstract_voter`: The first voter.
- `voter_2::Abstract_voter`: The second voter.

# Returns
- `distance::Real`: The distance between the given voters.
"""
get_distance(voter_1::T, voter_2::T) where {T<:Abstract_voter} = get_distance(get_opinion(voter_1), get_opinion(voter_2))
function get_distance(opinion_1::Vector{T}, opinion_2::Vector{T}) where {T<:Real}
	return Distances.evaluate(Distances.Cityblock(), opinion_1, opinion_2)
end

"""
	get_distance(voter::Abstract_voter, voters::Vector{Abstract_voter})::Vector{Real}

Returns the distances between the given voter and the given voters.

# Arguments
- `voter::Abstract_voter`: The voter.
- `voters::Vector{Abstract_voter}`: The voters.

# Returns
- `distances::Vector{Real}`: The distances between the given voter and the given voters.
"""
get_distance(voter::T, voters::Vector{T}) where {T<:Abstract_voter} = get_distance(get_opinion(voter), get_opinion(voters))
function get_distance(opinion::Vector{T}, opinions::Vector{Vector{T}}) where {T<:Real}
	return Distances.colwise(Distances.Cityblock(), opinion, reduce(hcat, opinions))
end

"""
	get_distance(voters::Vector{Abstract_voter})::Matrix{Real}

Returns the distances between the given voters.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters.

# Returns
- `distances::Matrix{Real}`: The distances between the given voters.
"""
get_distance(voters::Vector{T}) where {T<:Abstract_voter} = get_distance(get_opinion(voters))
function get_distance(opinions::Vector{Vector{T}}) where {T<:Real}
	return Distances.pairwise(Distances.Cityblock(), reduce(hcat, opinions), dims=2)
end

"""
	get_distance(voters_1::Vector{Abstract_voter}, voters_2::Vector{Abstract_voter})::Matrix{Real}

Returns the distances between the given voters.

# Arguments
- `voters_1::Vector{Abstract_voter}`: The first voters.
- `voters_2::Vector{Abstract_voter}`: The second voters.

# Returns
- `distances::Matrix{Real}`: The distances between the given voters.
"""
get_distance(voters_1::Vector{T}, voters_2::Vector{T}) where {T<:Abstract_voter} = get_distance(get_opinion(voters_1), get_opinion(voters_2))
function get_distance(opinions_1::Vector{Vector{T}}, opinions_2::Vector{Vector{T}}) where {T<:Real}
	return Distances.pairwise(Distances.Cityblock(), reduce(hcat, opinions_1), reduce(hcat, opinions_2), dims=2)
end

"""
	get_avg_distance(voter_1::Abstract_voter, voter_2::Abstract_voter)::Real

Returns the average distance between the given voters.

# Arguments
- `voter_1::Abstract_voter`: The first voter.
- `voter_2::Abstract_voter`: The second voter.

# Returns
- `distance::Real`: The average distance between the given voters.
"""
get_avg_distance(voters_1::Vector{T}, voters_2::Vector{T}) where {T<:Abstract_voter} = get_avg_distance(get_opinion(voters_1), get_opinion(voters_2))
function get_avg_distance(opinions_1::Vector{Vector{T}}, opinions_2::Vector{Vector{T}}) where {T<:Real}
	distance_matrix = get_distance(opinions_1, opinions_2)
	return sum(distance_matrix) / length(distance_matrix)
end

"""
	get_avg_distance(voters::Vector{Abstract_voter})::Real

Returns the average distance between the given voters.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters.

# Returns
- `distance::Real`: The average distance between the given voters.
"""
get_avg_distance(voters::Vector{T}) where {T<:Abstract_voter} = get_avg_distance(get_opinion(voters))
function get_avg_distance(opinions::Vector{T}) where {T<:Real}
	distance_matrix = get_distance(opinions)
	n = size(distance_matrix, 1)
	return sum(distance_matrix) / (n * (n - 1))
end

"""
	get_median_distance(voters::Vector{Abstract_voter})::Real

Returns the median distance between the given voters.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters.

# Returns
- `distance::Real`: The median distance between the given voters.
"""
function get_median_distance(distance_matrix)
	n = size(distance_matrix, 1)
	return StatsBase.median(sort(vec(distance_matrix))[n+1:end])
end
