"""
	translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::Float64)

Translates the given value from the range [oldMin, oldMax] to the range [newMin, newMax].

# Arguments
- `oldMin::Float64`: The minimum value of the old range.
- `oldMax::Float64`: The maximum value of the old range.
- `newMin::Float64`: The minimum value of the new range.
- `newMax::Float64`: The maximum value of the new range.

# Returns
- `translated::Float64`: The translated value.
"""
function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::Float64)
	return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
end

"""
	translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::AbstractVector)

Translates the given vector from the range [oldMin, oldMax] to the range [newMin, newMax].

# Arguments
- `oldMin::Float64`: The minimum value of the old range.
- `oldMax::Float64`: The maximum value of the old range.
- `newMin::Float64`: The minimum value of the new range.
- `newMax::Float64`: The maximum value of the new range.

# Returns
- `translated::Vector{Float64}`: The translated vector.
"""
function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::AbstractVector)
	return (((oldValue .- oldMin) .* (newMax - newMin)) ./ (oldMax - oldMin)) .+ newMin
end

"""
	normalize(vect::AbstractVector{T}) where {T<:Number}

Normalizes the given vector to the range [0, 1].

# Arguments
- `vect::AbstractVector{T}`: The vector to normalize.

# Returns
- `normalized::Vector{Float64}`: The normalized vector.
"""
function normalize(vect::AbstractVector{T}) where {T<:Number}
	minV = minimum(vect)
	maxV = maximum(vect)
	return (vect .- minV) ./ (maxV - minV)
end

"""
	choose2(n::Integer)

Returns n choose 2

# Arguments
- `n::Integer`: The number to choose 2 from.

# Returns
- `n choose 2::Integer`: The result of n choose 2.
"""
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

"""
	get_random_vote(can_count)::Vote

Returns a random vote with the given number of candidates.

# Arguments
- `can_count::Integer`: The number of candidates.

# Returns
- `vote::Vote`: The random vote.
"""
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

"""
	to_string(vote::Vote)

Returns a string representation of the given vote.

# Arguments
- `vote::Vote`: The vote to convert to a string.

# Returns
- `string::String`: The string representation of the vote.
"""
function to_string(vote::Vote)
	string = []
	for bucket in vote
		push!(string, join(bucket, " "))
	end

	return join(string, " | ")
end

"""
	get_frequent_votes(votes::Vector{Vote}, n::Int64)

Returns the n most frequent votes.

# Arguments
- `votes::Vector{Vote}`: The votes to get the most frequent votes from.

# Returns
- `top_n::Vector{Tuple{Vote, Int}}`: The n most frequent votes.
"""
function get_frequent_votes(votes::Vector{Vote}, n::Int64)
	vote_counts = Dict()
	for vote in votes
		vote_counts[vote] = get(vote_counts, vote, 0) + 1
	end
	top_n = collect(sort(vote_counts, rev=true; byvalue=true))[1:min(length(vote_counts), n)]

	return top_n#[(to_string(vote), count) for (vote, count) in top_n]
end

"""
	aggregate_mean(xs, ys)

Aggregates the given x and y values by taking the mean of the y values for each x value.

# Arguments
- `xs::Vector{Int}`: The x values.
- `ys::Vector{Float64}`: The y values.

# Returns
- `x_values::Vector{Int}`: The aggregated x values.
- `y_values::Vector{Float64}`: The aggregated y values.
"""
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
