struct Candidate
    ID::Int64
    name::String
    party::Int64
end

struct Election
    parties::Vector{String}
    candidates::Vector{Candidate}
    votes::Vector{Vote}
end

function parse_data(path_data::String)
    ext = Symbol(lowercase(splitext(path_data)[2][2:end]))
 
    return parse_data(path_data, Val(ext))
end
 
parse_data(path_data::String, ext)::Election = throw(ArgumentError("Unsupported format of input data $ext. Supported: [toc, soi]"))

@kwdef struct Selection_config <: Config
    remove_candidates::Vector{Int64},

    rng::Random.MersenneTwister,
	sample_size::Int64
end

function select(election::Election, selection_config::Selection_config)::Election
    filtered_election, candidates = remove_candidates(election.votes, election.candidates, election.remove_candidates)
    
    election = filtered_election[StatsBase.sample(selection_config.rng, 1:length(filtered_election), selection_config.sample_size, replace=false)]
    
    return Election(election.party_names, candidates, election)
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
            push!(new_candidates, OpinionDiffusion.Candidate(can.ID, can.name, can.party))
        end
    end

    #candidates = deleteat!(copy(candidates), remove_candidates)

    return new_election, new_candidates
end