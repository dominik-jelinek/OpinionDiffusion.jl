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

function get_pos(voter::Spearman_voter, can)
	return get_opinion(voter)[can]
end

function spearman_weights(weighting_rate, can_count)
	weight_func = position -> (can_count - position)^weighting_rate

	weights = Vector{Float64}(undef, can_count)
	weights[1] = 0.0
	for i in 2:length(weights)
		weights[i] = weights[i - 1] + weight_func(i - 1) + 1
	end

	# normalize by max distance
	max_sp_distance = get_max_distance(can_count, weights)
	weights = weights ./ max_sp_distance

	eps=(weights[end] - weights[end - 1])/4
	return weights, eps
end

function get_max_distance(can_count, weights)
	a = Vote()
	b = Vote()
	for i in 1:can_count
		push!(a, Bucket([i]))
		push!(b, Bucket([can_count - i + 1]))
	end

	return get_distance(spearman_encoding(a, weights), spearman_encoding(b, weights))
end