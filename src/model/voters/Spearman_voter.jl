struct Spearman_voter <: Abstract_voter
	ID::Int64

	opinion::Vector{Float64} # pSP
	eps::Float64 # cPBO
	weights::Vector{Float64} # cPBO

	properties::Dict{String,Any}
end

@kwdef struct Spearman_voter_config <: Abstract_voter_config
	weighting_rate::Float64
end
name(config::Spearman_voter_config) = "Spearman voter"

"""
    init_voters(votes::Vector{Vote}, voter_config::Spearman_voter_config)::Vector{Spearman_voter}

Initializes voters from the given votes and voter_config.

# Arguments
- `votes::Vector{Vote}`: The votes to initialize the voters with.
- `voter_config::Spearman_voter_config`: The config to initialize the voters with.

# Returns
- `voters::Vector{Spearman_voter}`: The initialized voters.
"""
function init_voters(votes::Vector{Vote}, voter_config::Spearman_voter_config)::Vector{Spearman_voter}
	can_count = candidate_count(votes[1])
	weights, eps = spearman_weights(voter_config.weighting_rate, can_count)

	voters = Vector{Spearman_voter}(undef, length(votes))
	for (i, vote) in enumerate(votes)
		opinion = spearman_encoding(vote, weights)

		properties = Dict()
		voters[i] = Spearman_voter(i, opinion, eps, weights, properties)
	end

	return voters
end

"""
	spearman_encoding(vote, weights)

Encodes bucket ordered vote to spearman encoded space

# Arguments
- `vote::Vote`: The vote to encode.
- `weights::Vector{Float64}`: The weights to use for the encoding.

# Returns
- `opinion::Vector{Float64}`: The opinion of the given vote.
"""
function spearman_encoding(vote::Vote, weights)
	opinion = Vector{Float64}(undef, length(weights))

	i = 1
	for bucket in vote
		if length(bucket) == 1
			opinion[iterate(bucket)[1]] = weights[i]
		else
			mean = sum(weights[i:i+length(bucket)-1]) / length(bucket)

			for can in bucket
				opinion[can] = mean
			end
		end

		i += length(bucket)
	end

	return opinion
end

"""
    get_vote(voter::Spearman_voter)::Vote

Returns the vote of the given voter.

# Arguments
- `voter::Spearman_voter`: The voter to get the vote of.

# Returns
- `vote::Vote`: The vote of the given voter.
"""
function get_vote(voter::Spearman_voter)::Vote
	# sort indexes based on opinions
	opinion = get_opinion(voter)
	can_ranking = sortperm(opinion)
	sorted_scores = opinion[can_ranking]

	vote = Vote()
	# pre fill first bucket
	counter = 1
	push!(vote, Bucket([can_ranking[1]]))

	for i in 2:length(sorted_scores)
		# if the opinion about the next candidate is at most eps from the last it is added into the same bucket
		if (sorted_scores[i] - sorted_scores[i-1]) < voter.eps
			push!(vote[counter], can_ranking[i])
		else
			push!(vote, Bucket([can_ranking[i]]))
			counter += 1
		end
	end

	return vote
end

"""
    get_pos(voter::Spearman_voter, can::Int64)::Float64

Returns the position of the given candidate in the opinion of the given voter.

# Arguments
- `voter::Spearman_voter`: The voter to get the position of the candidate in.
- `can::Int64`: The candidate to get the position of.

# Returns
- `pos::Float64`: The position of the given candidate in the opinion of the given voter.
"""
function get_pos(voter::Spearman_voter, can)
	return get_opinion(voter)[can]
end

"""
    spearman_weights(weighting_rate, can_count)

Returns the weights to use for the spearman encoding.

# Arguments
- `weighting_rate::Float64`: The weighting rate to use.
- `can_count::Int64`: The number of candidates.

# Returns
- `weights::Vector{Float64}`: The weights to use for the spearman encoding.
"""
function spearman_weights(weighting_rate, can_count)
	weight_func = position -> (can_count - position)^weighting_rate

	weights = Vector{Float64}(undef, can_count)
	weights[1] = 0.0
	for i in 2:length(weights)
		# adding one adds virtual bucket between neighoring buckets in full can_ranking
		# as the number of buckets needs to be at least 2 * number of candidates - 1
		weights[i] = weights[i - 1] + weight_func(i - 1)# + 1
	end

	# normalize by max distance
	max_sp_distance = get_max_distance(can_count, weights)
	weights = weights ./ max_sp_distance

	# divide by 4 instead of 2 as the minimum distance between two buckets in full ranking is now 2
	eps=(weights[end] - weights[end - 1]) / 2
	return weights, eps
end

"""
    get_max_distance(can_count, weights)

Returns the maximum distance between two candidates in the spearman encoding.

# Arguments
- `can_count::Int64`: The number of candidates.
- `weights::Vector{Float64}`: The weights to use for the spearman encoding.

# Returns
- `max_distance::Float64`: The maximum distance between two candidates in the spearman encoding.
"""
function get_max_distance(can_count, weights)
	a = Vote()
	b = Vote()
	for i in 1:can_count
		push!(a, Bucket([i]))
		push!(b, Bucket([can_count - i + 1]))
	end

	return get_distance(spearman_encoding(a, weights), spearman_encoding(b, weights))
end
