#___ Kendall tau encoding ___________________________________________________

"""
Encodes vote into space of dimension canCount choose 2 
"""
function kendall_encoding(vote)
   can_count = length(vote)
   #inversion of preference
   inv_vote = zeros(length(vote))
   for i in 1:length(vote)
      if vote[i] == 0
         break
      end
      inv_vote[vote[i]] = i
   end

   opinion = Vector{Float64}(undef, choose2(can_count))
   counter = 1
   for i in 1:can_count-1
      for j in i+1:can_count
         opinion[counter] = get_penalty(inv_vote, i, j)
         counter += 1
      end
   end
   return opinion
end

function get_all_swaps(vote_1, vote_2)
   can_count = length(vote_1)
   swaps = Vector{Pair}()
   counter = 1
   diff = vote_1 - vote_2

   for i in 1:can_count-1
      for j in i+1:can_count
         if diff[counter] != 0
            push!(swaps, (i,j)=>diff[counter])
         end
         counter += 1
      end
   end
   return swaps
end

function get_penalty(inv_vote, can_1, can_2)
   penalty = 0
   #in order
   if inv_vote[can_1] < inv_vote[can_2]
      penalty = 1
   #out of order
   elseif inv_vote[can_1] > inv_vote[can_2]
      penalty = -1
   end
   
   return penalty
end

function infer_opinions(database)
   voter_count = size(database, 2)
   can_pairs_count = choose2(size(database, 1))
   opinions = zeros(Float64, can_pairs_count, voter_count)

   for (i, vote) in enumerate(eachcol(database))
      opinions[:, i] .= kendall_encoding(vote)
   end

   return opinions
end

#___ Spearmann encoding ___________________________________________________



function update_distances(opinions, distances, v, metric::Distances.Metric) #not used
   for i in 1:size(distances, 2)
       new_distance = metric(getCol(opinions, i), getCol(opinions, v))
       distances[i, v] = new_distance
       distances[v, i] = new_distance
   end
end