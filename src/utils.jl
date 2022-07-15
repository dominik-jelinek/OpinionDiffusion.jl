function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::Float64)
   return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin;
end

function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::AbstractVector)
   return (((oldValue .- oldMin) .* (newMax - newMin)) ./ (oldMax - oldMin)) .+ newMin;
end

function normalize(vect::AbstractVector{T}) where T <: Number
   minV = minimum(vect)
   maxV = maximum(vect)
   return (vect .- minV) ./ (maxV - minV)
end

#Returns n choose 2
function choose2(n::Integer)
   return n*(n-1)÷2
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

function last_log_idx(exp_dir)
   idx = -1
   for filename in readdir(exp_dir)
      val = parse(Int64, chop(split(filename, "_")[end], tail=5))
      if  val > idx
         idx = val
      end
   end
   
   return idx
end

function get_random_vote(can_count) :: Vote
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
         push!(vote[curr_bin - skip_bins], sorted_cans[i])
      end
   end

   return vote
end

function filter_candidates(election, candidates, remove_candidates, can_count)
	if length(remove_candidates) == 0
		return election, candidates
	end
	# calculate candidate index offset dependant 
	adjust = zeros(can_count)
	for i in 1:length(remove_candidates) - 1
		adjust[remove_candidates[i] + 1:remove_candidates[i + 1]-1] += fill(i, remove_candidates[i + 1] - remove_candidates[i] - 1)
	end
	adjust[remove_candidates[end] + 1:end] += fill(length(remove_candidates), can_count - remove_candidates[end])

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
			push!(new_candidates, OpinionDiffusion.Candidate(can.name, can.party))
		end
	end
	
	#candidates = deleteat!(copy(candidates), remove_candidates)
	
	return new_election, new_candidates
end