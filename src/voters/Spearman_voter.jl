struct Spearman_voter <: Abstract_voter
    ID::Int64
    opinion::Vector{Float64}

    openmindedness::Float64
    stubbornness::Float64
end

@kwdef struct Spearman_voter_init_config <: Abstract_voter_init_config
    weight_func::Function
    openmindedness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
    stubbornness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
end

@kwdef struct Spearman_voter_diff_config <: Abstract_voter_diff_config
	attract_proba::Float64
	change_rate::Float64
    normalize_shifts::Union{Nothing, Tuple{Bool, Float64, Float64}}
end

function Spearman_voter(ID, vote, weights, max_distance, openmindedness_distr::Distributions.ContinuousUnivariateDistribution, stubbornness_distr::Distributions.ContinuousUnivariateDistribution)
    opinion = spearman_encoding(vote, weights, max_distance)
    openmindedness = rand(openmindedness_distr)
    stubbornness = rand(stubbornness_distr)

    return Spearman_voter(ID, opinion, openmindedness, stubbornness)
end

function init_voters(election, can_count, voter_config::Spearman_voter_init_config)
    weights = Vector{Float64}(undef, can_count)
    weights[1] = 0.0
    for i in 2:length(weights)
        weights[i] = weights[i - 1] + voter_config.weight_func(i - 1)
    end
    openmindedness_distr = Distributions.Truncated(voter_config.openmindedness_distr, 0.0, 1.0)
    stubbornness_distr = Distributions.Truncated(voter_config.stubbornness_distr, 0.0, 1.0)

    a = Vector{Vector{Int64}}()
    b = Vector{Vector{Int64}}()
    for i in 1:m
        push!(a, [i])
        push!(b, [m - i + 1])
    end
    max_distance = get_distance(spearman_encoding(a, weights), spearman_encoding(b, weights))
    
    voters = Vector{Spearman_voter}(undef, length(election))
    for (i, vote) in enumerate(election)
        voters[i] = Spearman_voter(i, vote, weights, max_distance, openmindedness_distr, stubbornness_distr)
    end

    return voters
end

"""
    spearman_encoding(vote, weights)

Encodes bucket ordered vote to spearman encoded space
"""
function spearman_encoding(vote::Vector{Vector{Int64}}, weights, max_distance=nothing)
    can_count = length(weights)

    opinion = Vector{Float64}(undef, can_count)
    mean = (weights[end] + weights[length(vote)-1]) / 2
    
    for (i, bucket) in enumerate(vote)
        if (length(bucket) == 1)
            opinion[bucket[1]] = weights[i]
        else
            for can in bucket
                opinion[can] = mean
            end
        end
    end
    
    return max_distance === nothing ? opinion : opinion ./ max_distance
end

function get_vote(voter::Spearman_voter; eps=0.01) :: Vector{Vector{Int64}}
    # sort indexes based on opinions
    can_ranking = sortperm(voter.opinion)
    sorted_scores = voter.opinion[can_ranking]

    vote = Vector{Vector{Int64}}()
    # pre fill first bucket
    counter = 1
    push!(vote, [can_ranking[1]])

    for i in 2:length(sorted_scores)
        # if the opinion about the next candidate is at most eps from the last it is added into the same bucket 
        if (sorted_scores[i] - sorted_scores[i-1]) < eps
            push!(vote[counter], can_ranking[i])
        else
            push!(vote, [can_ranking[i]])
            counter += 1
        end
    end

    return vote 
end

function get_pos(voter::Spearman_voter, can)
    return get_opinion(voter)[can]
end

function step!(self::Spearman_voter, voters, graph, can_count, voter_diff_config::Spearman_voter_diff_config)
    neighbors_ = neighbors(graph, self.ID)
    if length(neighbors_) == 0
        return
    end
        
    neighbor_id = neighbors_[rand(1:end)]
    neighbor = voters[neighbor_id]

    average_all!(self, neighbor, voter_diff_config.attract_proba, voter_diff_config.change_rate, voter_diff_config.normalize_shifts)
end

function average_all!(voter_1::Spearman_voter, voter_2::Spearman_voter, attract_proba, change_rate, normalize=nothing)
    shifts_1 = (voter_2.opinion - voter_1.opinion) / 2
    shifts_2 = shifts_1 .* (-1.0)
    
    if rand() > attract_proba
        #repel
        shifts_1, shifts_2 = shifts_2, shifts_1
    end

    if normalize !== nothing && normalize[1]
        shifts_1 = normalize_shifts(shifts_1, voter_1.opinion, normalize[2], normalize[3])
        shifts_2 = normalize_shifts(shifts_2, voter_2.opinion, normalize[2], normalize[3])
    end

    voter_1.opinion .+= shifts_1 * (1.0 - voter_1.stubbornness) * change_rate
    voter_2.opinion .+= shifts_2 * (1.0 - voter_2.stubbornness) * change_rate
end

function normalize_shifts(shifts::Vector{Float64}, opinion::Vector{Float64}, min_opin, max_opin)
    # decrease opinion changes that push candidates outside of [min_opin, max_opin] boundary
    #safeguard
    if max_opin < min_opin
        min_opin, max_opin = max_opin, min_opin
    end

    normalized = Vector{Float64}(undef, length(shifts))
    for i in 1:length(shifts)
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