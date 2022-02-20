
struct Kendall_voter <: Abstract_voter
   ID::Int64
   vote::Vector{Vector{Int64}}
   opinion::Vector{Float64}

   openmindedness::Float64
   stubborness::Float64
end
#=
struct Kendall_voter
   vote::Vector{Vector{Int64}}
   opinion::Vector{Float64}
end
function Kendall_voter(vote, can_count)
   opinion = kendall_encoding(vote, can_count)
   return Kendall_voter(vote, opinion)
end
=#
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

   opinion = Vector{Float64}(undef, choose2(can_count))
   counter = 1
   for can_1 in 1:can_count-1
      for can_2 in can_1+1:can_count
         opinion[counter] = get_penalty(inv_vote, can_1, can_2)
         counter += 1
      end
   end
   return opinion
end

function step!(self::Kendall_voter, voters, graph, can_count, voter_diff_config)
   neighbors_ = neighbors(graph, self.ID)
   if length(neighbors_) == 0
      return
   end

   neighbor_id = neighbors_[rand(1:end)]
   neighbor = voters[neighbor_id]

   if rand() <= voter_diff_config["attract_proba"]
      attract_flip!(self, neighbor, can_count)
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
   if get_distance(self, neighbor) == 0.0
      return
   end

   if rand() > self.stubborness
      swaps = get_feasible_swaps(self, neighbor, can_count)
      if length(swaps) == 0
         print(self, neighbor)
         error("No feasible swaps but non-zero distance")
         return
      end
   
      swap = swaps[rand(1:length(swaps))]
      self = apply_swap(swap, self, can_count)
   end
   
   if rand() > neighbor.stubborness
      swaps = get_feasible_swaps(neighbor, self, can_count)
      if length(swaps) == 0
         print(self, neighbor)
         error("No feasible swaps but non-zero distance")
         return      
      end
   
      swap = swaps[rand(1:length(swaps))]
      neighbor = apply_swap(swap, neighbor, can_count)
   end
end

function apply_swap(swap, voter, can_count)
   can, bucket_idx, type = swap
   bucket = voter.vote[bucket_idx]
   new_vote = deepcopy(voter.vote)
   
   if type == :unbucket_left
      new_opinion = unbucket_left(can, voter.opinion, bucket, can_count)

      delete!(new_vote[bucket_idx], can)
      insert!(new_vote, bucket_idx, [can])

   elseif type == :unbucket_right
      new_opinion = unbucket_right(can, voter.opinion, bucket, can_count)
    
      delete!(new_vote[bucket_idx], can)
      insert!(new_vote, bucket_idx + 1, [can])

   elseif type == :rebucket_left
      new_opinion = rebucket_left(can, voter.opinion, voter.vote[bucket_idx - 1], bucket, can_count)
      
      if length(new_vote[bucket_idx]) == 1
         deleteat!(new_vote, bucket_idx)
      else
         delete!(new_vote[bucket_idx], can)
      end
      
      push!(new_vote[bucket_idx - 1], can)
   else
      new_opinion = rebucket_right(can, voter.opinion, bucket, voter.vote[bucket_idx + 1], can_count)

      if length(new_vote[bucket_idx]) == 1
         deleteat!(new_vote, bucket_idx)
      else
         delete!(new_vote[bucket_idx], can)
      end
      
      push!(new_vote[bucket_idx + 1], can)
   end

   return Kendall_voter(voter.ID, new_vote, new_opinion, voter.openmindedness, voter.stubborness)
end

function get_feasible_swaps(u::Kendall_voter, v::Kendall_voter, can_count)
   # vector of new votes and opinions for voter u if we choose specific swap 
   feasible_swaps = Vector{Tuple{Int64, Float64, Symbol}}()
   
   if length(u.vote[1]) > 1
      append!(feasible_swaps, unbucket(u.opinion, v.opinion, u.vote, 1, can_count))
   end

   for i in 2:length(u.vote)
      #check for unbucket
      if length(u.vote[i]) > 1
         append!(feasible_swaps, unbucket(u.opinion, v.opinion, u.vote, i, can_count))
      end

      append!(feasible_swaps, rebucket(u.opinion, v.opinion, u.vote, i - 1, i, can_count))
   end

   return feasible_swaps
end

function unbucket(u_opinion, v_opinion, u_vote, bucket_idx, can_count)
   feasible_swaps = Vector{Tuple{Int64, Float64, Symbol}}()
   bucket = u_vote[bucket_idx]

   for can in bucket
      opinion = unbucket_right(can, u_opinion, bucket, can_count)
      change = get_distance(opinion, v_opinion) - get_distance(u_opinion, v_opinion)
      if change < 0.0
         push!(feasible_swaps, (can, bucket_idx, :unbucket_right))
      end

      opinion = unbucket_left(can, u_opinion, bucket, can_count)
      change = get_distance(opinion, v_opinion) - get_distance(u_opinion, v_opinion)
      if change < 0.0
         push!(feasible_swaps, (can, bucket_idx, :unbucket_left))
      end
   end

   return feasible_swaps
end

function unbucket_right(can, opinion, bucket, can_count)
   new_opinion = deepcopy(opinion)
   for i in 1:length(bucket)
      if bucket[i] != can
         new_opinion[get_index(bucket[i], can, can_count)] = can < bucket[i] ? 1.0 : 0.0
      end
   end

   return new_opinion
end

function unbucket_left(can, opinion, bucket, can_count)
   new_opinion = deepcopy(opinion)
   for i in 1:length(bucket)
      if bucket[i] != can
         new_opinion[get_index(bucket[i], can, can_count)] = can < bucket[i] ? 0.0 : 1.0
      end
   end

   return new_opinion
end

function rebucket(u_opinion, v_opinion, u_vote, r_bucket_idx, can_count)
   feasible_swaps = Vector{Tuple{Int64, Float64, Symbol}}()
   l_bucket = u_vote[r_bucket_idx - 1]
   r_bucket = u_vote[r_bucket_idx]

   for can in l_bucket
      opinion = rebucket_right(can, u_opinion, l_bucket, r_bucket, can_count)
      change = get_distance(opinion, v_opinion) - get_distance(u_opinion, v_opinion)
      if change < 0.0
         push!(feasible_swaps, (can, r_bucket_idx - 1, :rebucket_right))
      end
   end

   for can in r_bucket
      opinion = rebucket_left(can, u_opinion, l_bucket, r_bucket, can_count)
      change = get_distance(opinion, v_opinion) - get_distance(u_opinion, v_opinion)
      if change < 0.0
         push!(feasible_swaps, (can, r_bucket_idx, :rebucket_left))
      end
   end

   return feasible_swaps
end

function rebucket_right(can, opinion, l_bucket, r_bucket, can_count)
   new_opinion = deepcopy(opinion)
   for i in 1:length(l_bucket)
      if l_bucket[i] != can
         new_opinion[get_index(l_bucket[i], can, can_count)] = can < l_bucket[i] ? 1.0 : 0.0
      end
   end

   for i in 1:length(r_bucket)
      new_opinion[get_index(r_bucket[i], can, can_count)] = 0.5
   end

   return new_opinion
end

function rebucket_left(can, opinion, l_bucket, r_bucket, can_count)
   new_opinion = deepcopy(opinion)

   for i in 1:length(l_bucket)
      new_opinion[get_index(l_bucket[i], can, can_count)] = 0.5
   end

   for i in 1:length(r_bucket)
      if r_bucket[i] != can
         new_opinion[get_index(r_bucket[i], can, can_count)] = can < r_bucket[i] ? 0.0 : 1.0
      end
   end

   return new_opinion
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

function invert_vote(vote, can_count)
   pos = Vector{Int64}(undef, can_count)

   for i in 1:length(vote)
      # iterate bucket
      for can in vote[i]
         pos[can] = i
      end
   end

   return pos
end

"""
Gets index of pair can_1 and can_2 in opinion
"""
function get_index(can_1, can_2, can_count)
   if can_1 > can_count || can_2 > can_count || can_1 == can_2
      throw(DomainError("can_1 or can 2"))
   end

   if can_1 > can_2
      can_1, can_2 = can_2, can_1
   end

   return sum(can_count-can_1+1 : can_count-1) + can_2 - can_1 
end

"""
Gets index of pair can_1 and can_2 in opinion
"""
function get_index1(can_1, can_2, can_count)
   if can_1 > can_count || can_2 > can_count || can_1 == can_2
      throw(DomainError("can_1 or can 2"))
   end

   if can_1 > can_2
      can_1, can_2 = can_2, can_1
   end

   return choose2(can_1 - 1) + (can_1 - 1) * (can_count - can_1 + 1) + can_2 - can_1
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
   penalty = 0.5
   
   if inv_vote[can_1] < inv_vote[can_2]
      # candidates are in order with numerical candidate ordering
      penalty = 0.0
   elseif inv_vote[can_1] > inv_vote[can_2]
      # candidates are out of order with numerical candidate ordering
      penalty = 1.0
   end
   
   return penalty
end