struct Candidate
	ID::Int64
	name::String
	party_ID::Int64
	party_name::String
end

get_ID(candidate::Candidate) = candidate.ID
get_name(candidate::Candidate) = candidate.name
get_party_ID(candidate::Candidate) = candidate.party_ID
get_party_name(candidate::Candidate) = candidate.party_name

struct Election
	candidates::Vector{Candidate}
	votes::Vector{Vote}
end

get_candidates(election::Election) = election.candidates
get_votes(election::Election) = election.votes

@kwdef struct Sampling_config <: Abstract_config
	rng_seed::UInt32=rand(UInt32)
	sample_size::Int64
end

@kwdef struct Election_config <: Abstract_config
	data_path::String
	remove_candidate_ids::Vector{Int64} = []
	sampling_config::Union{Sampling_config, Nothing} = nothing
end

"""
	init_election(config::Election_config)

Initializes an election from the given config. The election is loaded from the given data_path and the candidates with the given IDs are removed. If a sampling_config is given, the election is sampled with the given sampling_config.

# Arguments
- `config::Election_config`: The config to initialize the election from.

# Returns
- `election::Election`: The initialized election.
"""
function init_election(config::Election_config)
	election = parse_data(config.data_path)

	if length(config.remove_candidate_ids) > 0
		election = remove_candidates(election, config.remove_candidate_ids)
	end

	if config.sampling_config !== nothing
		election = sample(election, config.sampling_config)
	end

	return election
end

"""
	remove_candidates(election::Election, candidate_ids::Vector{Int64})::Election

Removes the candidates with the given IDs from the given election.

# Arguments
- `election::Election`: The election to remove the candidates from.
- `candidate_ids::Vector{Int64}`: The IDs of the candidates to remove.

# Returns
- `election::Election`: The election without the removed candidates.
"""
function remove_candidates(election::Election, candidate_ids::Vector{Int64})::Election
	filtered_votes, filtered_candidates = remove_candidates(election.votes, election.candidates, candidate_ids)

	return Election(filtered_candidates, filtered_votes)
end

"""
	remove_candidates(votes::Vector{Vote}, candidates::Vector{Candidate}, candidate_ids::Vector{Int64})::Tuple{Vector{Vote}, Vector{Candidate}}

Removes the candidates with the given IDs from the given votes and candidates.

# Arguments
- `votes::Vector{Vote}`: The votes to remove the candidates from.
- `candidates::Vector{Candidate}`: The candidates to remove the candidates from.
- `candidate_ids::Vector{Int64}`: The IDs of the candidates to remove.

# Returns
- `votes::Vector{Vote}`: The votes without the removed candidates.
- `candidates::Vector{Candidate}`: The candidates without the removed candidates.
"""
function remove_candidates(votes::Vector{Vote}, candidates::Vector{Candidate}, candidate_ids::Vector{Int64})
	if length(candidate_ids) == 0
		return votes, candidates
	end

	can_count = length(candidates)

	# calculate candidate index offset dependant
	adjust = zeros(can_count)
	for i in 1:length(candidate_ids)-1
		adjust[candidate_ids[i]+1:candidate_ids[i+1]-1] += fill(i, candidate_ids[i+1] - candidate_ids[i] - 1)
	end
	adjust[candidate_ids[end]+1:end] += fill(length(candidate_ids), can_count - candidate_ids[end])

	#copy election without the filtered out candidates
	new_election = Vector{Vote}()
	for vote in votes
		new_vote = Vote()
		for bucket in vote
			new_bucket = Bucket()

			for can in bucket
				if can ∉ candidate_ids
					push!(new_bucket, can - adjust[can])
				end
			end

			if length(new_bucket) != 0
				push!(new_vote, new_bucket)
			end
		end

		# vote with one bucket ore less buckets contains no preferences
		if length(new_vote) > 1
			push!(new_election, new_vote)
		end
	end

	new_candidates = Vector{OpinionDiffusion.Candidate}()
	for (i, can) in enumerate(candidates)
		if i ∉ candidate_ids
			push!(new_candidates, OpinionDiffusion.Candidate(get_ID(can), get_name(can), get_party_ID(can), get_party_name(can)))
		end
	end

	#candidates = deleteat!(copy(candidates), remove_candidates)

	return new_election, new_candidates
end

"""
	sample(election::Election, sampling_config::Sampling_config)::Election

Samples the given election with the given sampling_config.

# Arguments
- `election::Election`: The election to sample.
- `sampling_config::Sampling_config`: The sampling_config to sample the election with.

# Returns
- `election::Election`: The sampled election.
"""
function sample(election::Election, sampling_config::Sampling_config)::Election
	rng = MersenneTwister(sampling_config.rng_seed)
	votes = get_votes(election)
	votes = votes[StatsBase.sample(rng, 1:length(votes), sampling_config.sample_size, replace=false)]

	return Election(get_candidates(election), votes)
end
