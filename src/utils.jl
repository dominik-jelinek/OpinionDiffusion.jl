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

Base.showerror(io::IO, ie::NotImplementedError) = print(io, "method $(ie.m) not implemented.")

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