notnothing(::Any) = true
notnothing(::Nothing) = false

function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::Float64)
   return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin;
end

function translateRange(oldMin::Float64, oldMax::Float64, newMin::Float64, newMax::Float64, oldValue::AbstractVector)
   return (((oldValue .- oldMin) .* (newMax - newMin)) ./ (oldMax - oldMin)) .+ newMin;
end

#Returns n choose 2
function choose2(n::Integer)
   return n*(n-1)÷2
end