struct Candidate
	ID::Int64
	name::String
	party_ID::Int64
end

get_ID(candidate::Candidate) = candidate.ID
get_name(candidate::Candidate) = candidate.name
get_party_ID(candidate::Candidate) = candidate.party_ID

struct Election
	party_names::Vector{String}
	candidates::Vector{Candidate}
	votes::Vector{Vote}
end

get_party_names(election::Election) = election.party_names
get_candidates(election::Election) = election.candidates
get_votes(election::Election) = election.votes

function parse_data(data_path::String)
	ext = Symbol(lowercase(splitext(data_path)[2][2:end]))

	return parse_data(data_path, Val(ext))
end

parse_data(data_path::String, ext)::Election = throw(ArgumentError("Unsupported format of input data $ext. Supported: [toc, soi]"))

@kwdef struct Selection_config <: Abstract_config
	remove_candidates::Vector{Int64}

	rng_seed::UInt32
	sample_size::Int64
end

function select(election::Election, selection_config::Selection_config)::Election
	filtered_votes, candidates = remove_candidates(election.votes, election.candidates, selection_config.remove_candidates)

	rng = MersenneTwister(selection_config.rng_seed)
	votes = filtered_votes[StatsBase.sample(rng, 1:length(filtered_votes), selection_config.sample_size, replace=false)]

	return Election(election.party_names, candidates, votes)
end

function remove_candidates(election, candidates, remove_candidates)
	can_count = length(candidates)

	if length(remove_candidates) == 0
		return election, candidates
	end
	# calculate candidate index offset dependant
	adjust = zeros(can_count)
	for i in 1:length(remove_candidates)-1
		adjust[remove_candidates[i]+1:remove_candidates[i+1]-1] += fill(i, remove_candidates[i+1] - remove_candidates[i] - 1)
	end
	adjust[remove_candidates[end]+1:end] += fill(length(remove_candidates), can_count - remove_candidates[end])

	#copy election without the filtered out candidates
	new_election = Vector{Vote}()
	for vote in election
		new_vote = Vote()
		for bucket in vote
			new_bucket = Bucket()

			for can in bucket
				if can ∉ remove_candidates
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
		if i ∉ remove_candidates
			push!(new_candidates, OpinionDiffusion.Candidate(get_ID(can), can.name, get_party_ID(can)))
		end
	end

	#candidates = deleteat!(copy(candidates), remove_candidates)

	return new_election, new_candidates
end