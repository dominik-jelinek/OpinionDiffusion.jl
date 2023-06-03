struct Spearman_voter <: Abstract_voter
    ID::Int64

    opinion::Vector{Float64} # pSP
    eps::Float64 # cPBO

    properties::Dict{String,Any}
end

@kwdef struct Spearman_voter_init_config <: Abstract_voter_init_config
    weights::Vector{Float64}
    eps::Float64
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

function init_voters(election, voter_config::Spearman_voter_init_config)

    voters = Vector{Spearman_voter}(undef, length(election))
    for (i, vote) in enumerate(election)
        opinion = spearman_encoding(vote, voter_config.weights)

        properties = Dict()
        voters[i] = Spearman_voter(i, opinion, voter_config.eps, properties)
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