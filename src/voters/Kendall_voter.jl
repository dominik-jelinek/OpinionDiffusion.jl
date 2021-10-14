
struct Kendall_voter <: Abstract_voter
   ID::Int64
   vote::Vector{Vector{Int64}}
   opinion::Vector{Int64}

   openmindedness::Float64
   stubborness::Float64
end

function get_vote(voter::Kendall_voter) :: Vector{Int}
   return voter.vote
end

"""
Encodes vote into space of dimension canCount choose 2 
"""
function kendall_encoding(vote::Vector{Vector{Int64}}, can_count)
   #inversion of preference
   inv_vote = zeros(can_count)
   for (i, pos) in enumerate(vote)
      for can in pos
         inv_vote[can] = i
      end
   end

   opinion = Vector{Int64}(undef, choose2(can_count))
   counter = 1
   for can_1 in 1:can_count-1
      for can_2 in can_1+1:can_count
         opinion[counter] = get_penalty(inv_vote, can_1, can_2)
         counter += 1
      end
   end
   return opinion
end

function get_all_swaps(voter_1, voter_2)
   swaps = Vector{Pair}()
   diff = voter_1.opinion - voter_2.opinion

   for (i, val) in enumerate(diff)
      if val != 0
         push!(swaps, i=>val)
      end
   end

   return swaps
end

function get_index(can_1, can_2, can_count)
   return sum(can_count-can_1+1 : can_count-1) + can_2 - can_1 
end

function get_candidates(index, can_count)
   for i in 1:can_count-1
      if index - can_count + i <= 0
         return (i, i + index)
      else
         index = index - can_count + i
      end
   end
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

function step!(self::Kendall_voter, voters, graph, voter_diff_config)
   neighbors_ = neighbors(graph, self.ID)
   if length(neighbors_) == 0
      return
   end

   neighbor_id = neighbors_[rand(1:end)]
   neighbor = voters[neighbor_id]
   if rand() <= voter_diff_config["attract_proba"]
      attract_flip!(self, neighbor)
   else
      repel_flip!(self, neighbor)
   end
   
end
#=
Choose pair of candidates that is flipped and make a one step of candidate position towards the other voter for each voter in pair.
   There are multiple ways how to do such an operation as there can be more than one candidate at each position in the preference.
   [[1], [2, 3], [4]]    [[1], [2], [3], [4]] pair 2, 3
   [[1], [2], [3], [4]]  [[1], [2], [4], [3]] pair 3, 4
   
=#
function attract_flip!(self, neighbor, can_count)
   swaps = get_all_swaps(self.opinion, neighbor.opinion)
   if length(swaps) == 0
      return
   end

   index, val = swaps[rand(1:length(swaps))]
   can_1, can_2 = get_candidates(index, can_count)
   
   # check if candidates are in wrong order
   if val > 0.0
      can_1, can_2 = can_2, can_1
   end

   if abs(val) == 2.0
      # self moves his preference towards v by one step
      if rand() < 0.5
         swapRight!(self, can_1)
      else
         swapLeft!(self, can_2)
      end
   elseif self.opinion[index] == 0
      # self makes idea about more favorable candidate in v
      if val == 1.0
         if rand() < 0.5
            swapRight!(self, can_1)
         else
            swapLeft!(self, can_2)
         end
      else
         if rand() < 0.5
            swapRight!(self, can_2)
         else
            swapLeft!(self, can_1)
         end
      end
   else # neighbor.opinion[index] == 0
      # neighbor hasn't got a preference for can1 and can2
      return
   end

   idx = Nothing
   for (i, bucket) in enumerate(self.vote)
      if can_1 in bucket
         idx = i
         break
      end
   end

end

function repel_flip!(voter_1, voter_2)

end

"""
Gets all swaps in between 2 preferences and chooses one.
The left(right) candidate in reversed pair is moved to the right(left) by one position
"""
function changeNeighbor!(database, encodedDB, v, neighbor)
   neighborPref = getCol(database, neighbor)
   
   # pick a swap
   swaps = getAllSwaps(getCol(encodedDB, v), neighborPref)
   if length(swaps) == 0
       return
   end
   (can1, can2), val = swaps[rand(1:length(swaps))]

   idxCan1 = getIndex(neighborPref, can1)
   idxCan2 = getIndex(neighborPref, can2)
   
   # check if candidates are in wrong order
   if idxCan1 > idxCan2
       can1, can2 = can2, can1
       idxCan1, idxCan2 = idxCan2, idxCan1
   end

   if abs(val) == 2
       # neighbor moves his preference towards v by one step
       if rand() < 0.5
           swapRight!(neighborPref, idxCan1)
       else
           swapLeft!(neighborPref, idxCan2)
       end
   elseif idxCan1 == idxCan2
       # neighbor makes idea about more favorable candidate in v
       if val == 1
           swapLeft!(neighborPref, idxCan1)
       else
           swapLeft!(neighborPref, idxCan2)
       end
   else # abs(val) == 1
       # v hasn't got a preference for can1 and can2
       return
   end
end

function changeNeighbor2!(database, encodedDB, v, neighbor, weights)
   # pick random candidate
   can = rand(1:size(database, 1))

   prefV = getCol(database, v)
   prefNei = getCol(database, neighbor)

   idxCanV = getIndex(prefV, can)
   idxCanNei = getIndex(prefNei, can)
   if prefV[idxCanV] == 0
       return # no swap, unknown preference in v
   end

   encodedNei = getCol(encodedDB, neighbor)
   # check if can is to the left or right in neighbor relative to v
   if idxCanV < idxCanNei
       # moving neighbor left
       canWeight = encodedNei[can]
       leftWeight = encodedNei[prefNei[idxCanNei - 1]]
       if rand() < 1 - (leftWeight + canWeight) / 2
           swapLeft!(prefNei, can, idxCanNei)
           encodedNei .= spearmanEncoding(prefNei, weights)
       end
   elseif idxCanV > idxCanNei
       # moving neighbor right
       canWeight = encodedNei[can]
       rightWeight = encodedNei[prefNei[idxCanNei + 1]]
       if rand() < 1 - (canWeight + rightWeight) / 2
           swapRight!(prefNei, idxCanNei)
           encodedNei .= spearmanEncoding(prefNei, weights)
       end
   else
       # no swap, unknown or the same position
   end
end

function changeNeighbor3!(database, encodedDB, v, neighbor, weights)
   # pick random candidates
   can1, can2 = StatsBase.sample(1:size(database, 1), 2)

   if encodedDB[can1, v] == encodedDB[can2, v]
       return # no swap, unknown preference in v
   end

   if encodedDB[can1, v] < encodedDB[can2, v]
       can1, can2 = can2, can1
   end
   
   prefV = getCol(database, v)
   prefNei = getCol(database, neighbor)
   idxCanV = getIndex(prefV, can)
   idxCanNei = getIndex(prefNei, can)

   encodedNei = getCol(encodedDB, neighbor)
   # check if can is to the left or right in neighbor relative to v
   if rand() < 0.5
       swapRight!(neighborPref, idxCan1)
   else
       swapLeft!(neighborPref, idxCan2)
   end
end

"""
Move candidate that is first out of can1 and can2 in the preference one position to the right
if can1 = 1 can2 = 3 (1,2,3) -> (2,1,3)
if can1 = 1 can2 = 2 (1,0,0) -> (2,1,0)
"""
function swapRight!(toChange, can2, idxCan1)
   # can1 or can2 must be in toChange

   # move to right
   if toChange[idxCan1 + 1] == 0
       toChange[idxCan1], toChange[idxCan1 + 1] = can2, toChange[idxCan1]
   else
       toChange[idxCan1], toChange[idxCan1 + 1] = toChange[idxCan1 + 1], toChange[idxCan1]
   end
end

"""
Move candidate that is further in the preference one position to the left
if can1 = 1 can2 = 3 (1,2,3) -> (2,3,1)
if can1 = 1 can2 = 3 (1,2,0) -> (1,2,3)
"""
function swapLeft!(toChange, can, idxCan)
   # can1 or can2 must be in toChange and idxCan1 < idxCan2

   # move to left
   if toChange[idxCan] == 0
       toChange[idxCan] = can
   else
       toChange[idxCan - 1], toChange[idxCan] = toChange[idxCan], toChange[idxCan - 1]
   end
end