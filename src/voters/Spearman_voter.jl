struct Spearman_voter <: Abstract_voter
    ID::Int64

    opinion::Vector{Float64} # cPBO

    openmindedness::Float64 # graph
    stubbornness::Float64 # step
end

@kwdef struct Spearman_voter_init_config <: Abstract_voter_init_config
    weights::Vector{Float64}
    openmindedness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
    stubbornness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
end

@kwdef struct Spearman_voter_diff_config <: Abstract_voter_diff_config
	attract_proba::Float64
	change_rate::Float64
    normalize_shifts::Union{Nothing, Tuple{Bool, Float64, Float64}}
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

function init_voters(election, can_count, voter_config::Spearman_voter_init_config; rng=Random.GLOBAL_RNG)
    openmindedness_distr = Distributions.Truncated(voter_config.openmindedness_distr, 0.0, 1.0)
    stubbornness_distr = Distributions.Truncated(voter_config.stubbornness_distr, 0.0, 1.0)
    
    voters = Vector{Spearman_voter}(undef, length(election))
    for (i, vote) in enumerate(election)
        opinion = spearman_encoding(vote, voter_config.weights)
        openmindedness = rand(rng, openmindedness_distr)
        stubbornness = 0.5#rand(rng, stubbornness_distr)
        voters[i] = Spearman_voter(i, opinion, openmindedness, stubbornness)
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
            mean = sum(weights[i:i + length(bucket) - 1]) / length(bucket)

            for can in bucket
                opinion[can] = mean
            end
        end

        i += length(bucket)
    end
    
    return opinion
end

function get_vote(voter::Spearman_voter; eps=0.005) :: Vote
    # sort indexes based on opinions
    can_ranking = sortperm(voter.opinion)
    sorted_scores = voter.opinion[can_ranking]

    vote = Vote()
    # pre fill first bucket
    counter = 1
    push!(vote, Bucket([can_ranking[1]]))

    for i in 2:length(sorted_scores)
        # if the opinion about the next candidate is at most eps from the last it is added into the same bucket 
        if (sorted_scores[i] - sorted_scores[i-1]) < eps
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

function step!(self::Spearman_voter, model, voter_diff_config::Spearman_voter_diff_config; rng=Random.GLOBAL_RNG)
    voters = get_voters(model)
    social_network = get_social_network(model)
    neighbors_ = neighbors(social_network, self.ID)
    
    if length(neighbors_) == 0
        return []
    end
        
    neighbor_id = neighbors_[rand(rng, 1:end)]
    neighbor = voters[neighbor_id]

    return average_all!(self, neighbor, voter_diff_config.attract_proba, voter_diff_config.change_rate, voter_diff_config.normalize_shifts; rng=rng)
end

function average_all!(voter_1::Spearman_voter, voter_2::Spearman_voter, attract_proba, change_rate, normalize=nothing; rng=Random.GLOBAL_RNG)
    shifts_1 = (voter_2.opinion - voter_1.opinion) / 2
    shifts_2 = shifts_1 .* (-1.0)
    
    method = "attract"
    if rand(rng) > attract_proba
        #repel
        method = "repel"
        shifts_1, shifts_2 = shifts_2, shifts_1
    end

    if normalize !== nothing && normalize[1]
        shifts_1 = normalize_shifts(shifts_1, voter_1.opinion, normalize[2], normalize[3])
        shifts_2 = normalize_shifts(shifts_2, voter_2.opinion, normalize[2], normalize[3])
    end

    cp_1 = deepcopy(voter_1)
    cp_2 = deepcopy(voter_2)
    voter_1.opinion .+= shifts_1 * (1.0 - voter_1.stubbornness) * change_rate
    voter_2.opinion .+= shifts_2 * (1.0 - voter_2.stubbornness) * change_rate
    return [Action(method, (voter_2.ID, voter_1.ID), cp_1, voter_1), Action(method, (voter_1.ID, voter_2.ID), cp_2, voter_2)]
end

function normalize_shifts(shifts::Vector{Float64}, opinion::Vector{Float64}, min_opin, max_opin)
    # decrease opinion changes that push candidates outside of [min_opin, max_opin] boundary
    #safeguard
    if max_opin < min_opin
        min_opin, max_opin = max_opin, min_opin
    end

    normalized = Vector{Float64}(undef, length(shifts))
    for i in eachindex(shifts)
        normalized[i] = normalize_shift(shifts[i], opinion[i], min_opin, max_opin)
    end

    return normalized
end

function normalize_shift(shift::Float64, can_opinion::Float64, min_opin, max_opin)
    if shift == 0.0 || min_opin <= can_opinion || can_opinion <= max_opin
        return shift
    end  

    return shift * (sign(shift) == 1.0 ? 2^(-can_opinion + max_opin) : 2^(can_opinion - min_opin))
end