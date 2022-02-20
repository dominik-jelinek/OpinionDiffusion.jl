notnothing(::Any) = true
notnothing(::Nothing) = false

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

function get_random_vote(can_count) :: Vector{Vector{Int64}}
   # assign candidates to random bins
   bins = rand(1:can_count, can_count)
   sorted_cans = sortperm(bins)
   sorted_bins = bins[sorted_cans]
 
   vote = Vector{Vector{Int64}}()
   curr_bin = 0
   skip_bins = 0
   for i in 1:length(sorted_cans)
      if curr_bin != sorted_bins[i]
         # lower bin index that candidates are assigned to as there might be bins without any candidates
         if sorted_bins[i] - curr_bin > 1
            skip_bins += sorted_bins[i] - curr_bin - 1
         end
         
         curr_bin = sorted_bins[i]
         push!(vote, [sorted_cans[i]])
      else
         push!(vote[curr_bin - skip_bins], sorted_cans[i])
      end
   end

   return vote
end