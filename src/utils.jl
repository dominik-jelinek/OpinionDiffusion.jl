function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::Float64)
    return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
end

function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::AbstractVector)
    return (((oldValue .- oldMin) .* (newMax - newMin)) ./ (oldMax - oldMin)) .+ newMin
end

function normalize(vect::AbstractVector{T}) where {T<:Number}
    minV = minimum(vect)
    maxV = maximum(vect)
    return (vect .- minV) ./ (maxV - minV)
end

#Returns n choose 2
function choose2(n::Integer)
    return n * (n - 1) ÷ 2
end

"""
   NotImplementedError{M}(m)
`Exception` thrown when a method from Abstract struct is not implemented in coresponding subtype.
"""
struct NotImplementedError{M} <: Exception
    m::M
    NotImplementedError(m::M) where {M} = new{M}(m)
end

Base.showerror(io::IO, ie::NotImplementedError) = print(io, "Method $(ie.m) not implemented.")

function get_random_vote(can_count)::Vote
    # assign candidates to random bins
    bins = rand(1:can_count, can_count)
    sorted_cans = sortperm(bins)
    sorted_bins = bins[sorted_cans]

    vote = Vote()
    curr_bin = 0
    skip_bins = 0
    for i in 1:length(sorted_cans)
        if curr_bin != sorted_bins[i]
            # lower bin index that candidates are assigned to as there might be bins without any candidates
            if sorted_bins[i] - curr_bin > 1
                skip_bins += sorted_bins[i] - curr_bin - 1
            end

            curr_bin = sorted_bins[i]
            push!(vote, Bucket(sorted_cans[i]))
        else
            push!(vote[curr_bin-skip_bins], sorted_cans[i])
        end
    end

    return vote
end

function to_string(vote::Vote)
    string = []
    for bucket in vote
        push!(string, join(bucket, " "))
    end

    return join(string, " | ")
end

function get_frequent_votes(votes::Vector{Vote}, n::Int64)
    vote_counts = Dict()
    for vote in votes
        vote_counts[vote] = get(vote_counts, vote, 0) + 1
    end
    top_n = collect(sort(vote_counts, rev=true; byvalue=true))[1:min(length(vote_counts), n)]

    return top_n#[(to_string(vote), count) for (vote, count) in top_n]
end

function aggregate_mean(xs, ys)
    sums = Dict{Int,Float64}()
    counts = Dict{Int,Int}()

    for (x, y) in zip(xs, ys)
        # Update sums and counts dictionaries
        if haskey(sums, x)
            sums[x] += y
            counts[x] += 1
        else
            sums[x] = y
            counts[x] = 1
        end
    end

    # Calculate averages
    averages = Dict{Int,Float64}()
    for (x, sum) in sums
        averages[x] = sum / counts[x]
    end
    pairs = sort(collect(averages), by=x -> x[1])
    return [x for (x, _) in pairs], [x for (_, x) in pairs]
end