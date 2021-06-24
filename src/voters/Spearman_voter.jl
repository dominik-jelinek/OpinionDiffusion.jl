struct Spearman_voter <: Abstract_voter
    ID::Int64
    opinion::Vector{Float64}

    openmindedness::Float64
    stubbornness::Float64
end

function Spearman_voter(ID, vote, weights, openmindedness_distr::ContinuousUnivariateDistribution, stubbornness_distr::ContinuousUnivariateDistribution)
    opinion = spearman_encoding(vote, weights)
    openmindedness = rand(openmindedness_distr)
    stubbornness = rand(stubbornness_distr)

    return Spearman_voter(ID, opinion, openmindedness, stubbornness)
end

function get_vote(voter::Spearman_voter) :: Vector{Vector{Int64}} # NOT TESTED
    can_ranking = sortperm(voter.opinion)
    sorted_scores = voter.opinion[can_ranking]
    
    vote = Vector{Vector{Int64}}()
    counter = 1
    push!(vote, [can_ranking[1]])
    for i in 2:length(sorted_scores)
        if sorted_scores[i-1] == sorted_scores[i]
            push!(vote[counter], can_ranking[i])
        else
            push!(vote, [can_ranking[i]])
            counter += 1
        end
    end
    return vote 
end

function spearman_encoding(vote::Vector{Vector{Int64}}, weights)
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
    
    return opinion ./ sum(opinion)
end
#=
function spearman_encoding(vote, weights)
    magic = -42.0
    opinion = fill(magic, length(vote))

    i = 1
    while i <= length(vote) && vote[i] != 0
       opinion[vote[i]] = weights[i]
       i += 1
    end
    if i == length(vote) + 1
       return opinion
    end
 
    #if the last in ranking is 0 we know its exact weight
    mean = (i == length(vote) && vote[i] == 0) ? weights[length(vote)] : (weights[length(vote)] + weights[i-1]) / 2
    
    for pos in 1:length(opinion)
       if opinion[pos] == magic
          opinion[pos] = mean
       end
    end
    return opinion
end
=#
function step!(self::Spearman_voter, voters, graph, voter_diff_config)
    neighbors_ = neighbors(graph, self.ID)
    if length(neighbors_) == 0
        return
    end
        
    neighbor_id = neighbors_[rand(1:end)]
    neighbor = voters[neighbor_id]

    if voter_diff_config["method"] == "averageOne"
        average_one!(self, neighbor)
    elseif voter_diff_config["method"] == "averageAll"
        average_all!(self, neighbor)
    else
        error("Unknown vertex diffusion method, [averageOne | averageAll]")
    end
end

function average_all!(voter_1::Spearman_voter, voter_2::Spearman_voter)
    distance = (voter_1.opinion - voter_2.opinion) / 2
        
    if rand() < 0.5 # could be a parameter
        # attract
        voter_1.opinion .-= distance * (1 - voter_1.stubbornness)
        voter_2.opinion .+= distance * (1 - voter_2.stubbornness)
    else
        # repel
        voter_1.opinion .+= distance * (1 - voter_1.stubbornness)
        voter_2.opinion .-= distance * (1 - voter_2.stubbornness)
    end
end

function average_one!(voter_1::Spearman_voter, voter_2::Spearman_voter)
    can = rand(1:length(voter_1.opinion))
    distance = (voter_1.opinion[can] - voter_2.opinion[can]) / 2
        
    if rand() < 0.5 # could be a parameter
        # attract
        voter_1.opinion[can] -= distance * (1 - voter_1.stubbornness)
        voter_2.opinion[can] += distance * (1 - voter_2.stubbornness)
    else
        # repel
        voter_1.opinion[can] += distance * (1 - voter_1.stubbornness)
        voter_2.opinion[can] -= distance * (1 - voter_2.stubbornness)
    end
end