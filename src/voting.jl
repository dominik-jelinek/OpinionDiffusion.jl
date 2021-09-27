function drawElectionResult(candidates, parties, result)
    groupedbar(
        result,
        title = "Election results",
        ylabel="Num. of votes",
        xlabel="Position in the preference",
        bar_position = :stack,
        bar_width = 0.7,
        xticks =(1:length(candidates)),
        label = [candidate.name * "-" * parties[candidate.party] for candidate in candidates],
        yformatter = :plain
    )
end

function plurality_voting(votes::Vector{Vector{Vector{Int64}}}, can_count,  normalize::Bool)
    result = zeros(Float64, can_count)

    for vote in votes
        for can in vote[1]
            result[can] += 1 / length(vote[1])
        end
    end

    return normalize ? (result/sum(result))*100 : result
end

function borda_voting(votes::Vector{Vector{Vector{Int64}}}, can_count, normalize::Bool)
    result = zeros(Float64, can_count)
    
    for vote in votes
        points = can_count
        for bucket in vote
            score = StatsBase.mean((points - length(bucket)):points-1)
            points -= length(bucket)
            
            for can in bucket
                result[can] += score
            end
        end
    end

    return normalize ? (result/sum(result))*100 : result
end

function copeland_voting(votes::Vector{Vector{Vector{Int64}}}, can_count)
    result = zeros(Float64, can_count, can_count)
    alpha = 0.5

    for vote in votes
        #iterate buckets in vote
        for (i, bucket) in enumerate(vote)
            #iterate candidates inside bucket
            for (j, can) in enumerate(bucket)
                #assign alpha score for succesors inside of the bucket
                for k in j+1:length(bucket)
                    result[can, bucket[k]] += alpha
                    result[bucket[k], can] += alpha
                end

                for k in i+1:length(vote)
                    for later_can in vote[k]
                        result[can, later_can] += 1
                    end
                end
            end
        end
    end

    scores = zeros(can_count)
    for i in 1:can_count
        for j in 1:can_count
            if result[i, j] > result[j, i]
                scores[i] += 1
            end
        end
    end
    return scores
end

# OLD ___________________________________________
#=
function getPluralityScores(electionResult, normalize::Bool)
    res = electionResult[1, :]
    if normalize
        return (res/sum(res))*100
    else
        return res
    end
end

function getBordaScores(electionResult, normalize::Bool)
    n = size(electionResult, 2)
    res = Array{Int,1}(undef, n)
    sum_ = 0

    for i in 1:n
        for j in 1:n
            sum_ += (n-j+1)*electionResult[j, i]
        end
        res[i] = sum_
        sum_ = 0
    end

    if normalize
        return (res/sum(res))*100
    else
        return res
    end
end


function getWinners(vect)
    n = length(vect)
    res = Array{Int,1}(undef, n)
    ordered = sort(vect)

    for i in 1:n
        for j in 1:length(ordered)
            if vect[i] == ordered[j]
                res[i] = n - j + 1
                break
            end
        end
    end
    return res
end

function getElectionResult(database)
    numOfCand = size(database, 1)
    #[pos, can]
    result = zeros(Int, numOfCand, numOfCand)
    for vote in eachcol(database)
        for i in eachindex(vote)
            if vote[i] == 0
                break
            end
            result[i, vote[i]] += 1
        end
    end
    return result
end
=#