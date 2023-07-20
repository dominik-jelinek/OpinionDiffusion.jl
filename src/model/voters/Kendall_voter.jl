struct Kendall_voter <: Abstract_voter
	ID::Int64

	opinion::Vector{Float64} #pKT
	vote::Vote # BO

	properties::Dict{String,Any}
end

@kwdef struct Kendall_voter_config <: Abstract_voter_config
end
name(config::Kendall_voter_config) = "Kendall voter"

"""
	init_voters(votes::Vector{Vote}, voter_config::Kendall_voter_config)::Vector{Kendall_voter}

Initializes voters from the given votes and voter_config.

# Arguments
- `votes::Vector{Vote}`: The votes to initialize the voters with.
- `voter_config::Kendall_voter_config`: The config to initialize the voters with.

# Returns
- `voters::Vector{Kendall_voter}`: The initialized voters.
"""
function init_voters(votes::Vector{Vote}, voter_config::Kendall_voter_config)::Vector{Kendall_voter}
	can_count = candidate_count(votes[1])

	voters = Vector{Kendall_voter}(undef, length(votes))
	for (i, vote) in enumerate(votes)
		opinion = kendall_encoding(vote, can_count)

		properties = Dict()
		voters[i] = Kendall_voter(i, opinion, vote, properties)
	end

	return voters
end

"""
	get_vote(voter::Kendall_voter)::Vote

Returns the vote of the given voter.

# Arguments
- `voter::Kendall_voter`: The voter to get the vote of.

# Returns
- `vote::Vote`: The vote of the given voter.
"""
function get_vote(voter::Kendall_voter)::Vote
	return voter.vote
end

"""
	get_pos(voter::Kendall_voter, can)::Int

Returns the position of the given candidate in the given voter's vote.

# Arguments
- `voter::Kendall_voter`: The voter to get the position of the candidate in.
- `can::Int`: The candidate to get the position of.

# Returns
- `pos::Int`: The position of the given candidate in the given voter's vote.
"""
function get_pos(voter::Kendall_voter, can)
	pos = 0
	for bucket in get_vote(voter)
		if can in bucket
			return pos + (length(bucket) + 1) / 2
		end

		pos += length(bucket)
	end

	return pos
end

"""
	kendall_encoding(vote::Vote, can_count)::Vector{Float64}

Encodes vote into space of dimension can_ount choose 2

# Arguments
- `vote::Vote`: The vote to encode.
- `can_count::Int`: The number of candidates.

# Returns
- `opinion::Vector{Float64}`: The encoded vote.
"""
function kendall_encoding(vote::Vote, can_count)
	inv_vote = invert_vote(vote, can_count)
	n = choose2(can_count)

	opinion = Vector{Float64}(undef, n)
	counter = 1
	for can_1 in 1:can_count-1
		for can_2 in can_1+1:can_count
			opinion[counter] = get_penalty(inv_vote[can_1], inv_vote[can_2], n)
			counter += 1
		end
	end

	return opinion
end
