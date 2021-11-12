
struct Kendall_voter <: Abstract_voter
   ID::Int64
   vote::Vector{Vector{Int64}}
   opinion::Vector{Int64}

   openmindedness::Float64
   stubborness::Float64
end

function Kendall_voter(ID, vote, can_count, openmindedness_distr::Distributions.ContinuousUnivariateDistribution, stubbornness_distr::Distributions.ContinuousUnivariateDistribution)
   opinion = kendall_encoding(vote, can_count)
   openmindedness = rand(openmindedness_distr)
   stubbornness = rand(stubbornness_distr)

   return Kendall_voter(ID, vote, opinion, openmindedness, stubbornness)
end

function init_voters(election, can_count, voter_config)
   openmindedness_distr = Distributions.Truncated(voter_config.openmindedness_distr, 0.0, 1.0)
   stubbornness_distr = Distributions.Truncated(voter_config.stubbornness_distr, 0.0, 1.0)

   voters = Vector{Kendall_voter}(undef, length(election))
   for (i, vote) in enumerate(election)
       voters[i] = Kendall_voter(i, vote, can_count, openmindedness_distr, stubbornness_distr)
   end

   return voters
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
Choose pair of candidates that is flipped and move them in both voters one step closer.
If the choosen pair is one distance away flip the coin who wins.
Each action of moving closer is approved by generating random number in the interval 0.0 - 1.0. 
If it is bigger than the stubborness of respective voter, it is executed.

   [[1], [2, 3]],   [[1], [2], [3]] == [[1], [2], [3]] (d:1->0)
   [[1], [2], [3]], [[1], [2, 3]]   == [[1], [2, 3]] (d:1->0)
   
   [[1], [2], [3]], [[1], [3], [2]] == [[1], [3], [2]] (d:2->0) | [[1], [2, 3]] (d:2->1)
   
   [[1], [2], [3]], [[2], [3], [1]] == [[2], [1], [3]] (d:4->2) | [[1, 2], [3]] (d:4->3)
   
   
=#
function attract_flip!(self, neighbor, can_count)
   swaps = get_all_swaps(self.opinion, neighbor.opinion)
   if length(swaps) == 0
      return
   end

   idx = swaps[rand(1:length(swaps))]
   can_1, can_2 = get_candidates(idx, can_count)

   if rand() > self.stubborness
      if self.opinion[idx] == -1
         fst_can, snd_can = can_2, can_1
      end

      if rand() < 0.5
         swap_fst!(self, fst_can, snd_can)
      else
         swap_snd!(self, fst_can, snd_can)
      end
   end
   
   if rand() > neighbor.stubborness
      if neighbor.opinion[idx] == -1
         fst_can, snd_can = can_2, can_1
      end

      if rand() < 0.5
         swap_right!(self, fst_can, snd_can)
      else
         swap_left!(self, fst_can, snd_can)
      end
   end
end

function repel_flip!(voter_1, voter_2)

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

# Utils _________________________________________________________________________________
"""
Gets all swapped candidate pairs indexes based on opinion difference.
"""
function get_all_swaps(voter_1, voter_2)
   swaps = Vector{Int64}()
   diff = voter_1.opinion - voter_2.opinion

   for i in 1:length(diff)
      if val != 0
         push!(swaps, i)
      end
   end

   return swaps
end

"""
Gets index of pair can_1 and can_2 in opinion
"""
function get_index(can_1, can_2, can_count)
   return sum(can_count-can_1+1 : can_count-1) + can_2 - can_1 
end

"""
Gets pair of candidates that represent value at index from opinion
"""
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
   # candidates are indistinguishable
   penalty = 0
   
   if inv_vote[can_1] < inv_vote[can_2]
      # candidates are in order with numerical candidate ordering
      penalty = 1
   elseif inv_vote[can_1] > inv_vote[can_2]
      # candidates are out of order with numerical candidate ordering
      penalty = -1
   end
   
   return penalty
end