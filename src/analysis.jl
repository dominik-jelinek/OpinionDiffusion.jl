Metrics(g::LightGraphs.AbstractGraph, distances::Matrix{Float64}, result::Matrix{Int}) = Metrics(
   sum(result) / LightGraphs.nv(g),
   sum(distances) / length(distances),
   maximum(distances),
   2*LightGraphs.ne(g) / (LightGraphs.nv(g) * (LightGraphs.nv(g) - 1)),
   2*LightGraphs.ne(g) / LightGraphs.nv(g),
   getPluralityScores(result, true),
   getBordaScores(result, true)
)

Metrics(g::LightGraphs.AbstractGraph, result::Matrix{Int}) = Metrics(
   result,
   sum(result) / LightGraphs.nv(g),
   0.0,
   0.0,
   2*LightGraphs.ne(g) / (LightGraphs.nv(g) * (LightGraphs.nv(g) - 1)),
   2*LightGraphs.ne(g) / LightGraphs.nv(g),
   getPluralityScores(result, true),
   getBordaScores(result, true)
)

function getDegreeDistribution(g, group)
   dict = Dict{Int, Int}()
   for i in 1:length(group)
      deg = degree(g, group[i])
      if haskey(dict, deg)
         dict[deg] = dict[deg] + 1
      else
         dict[deg] = 1
      end
   end
   return dict
end