
struct Kendall_voter <: Abstract_voter
   ID::Int64
   vote::Vector{Vector{Int64}}
   opinion::Vector{Float64}

   openmindedness::Float64
   stubborness::Float64
end

@kwdef struct Kendall_voter_init_config <: Abstract_voter_init_config
   openmindedness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
   stubbornness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
end

@kwdef struct Kendall_voter_diff_config <: Abstract_voter_diff_config
	attract_proba::Float64
end

function Kendall_voter(ID, vote, can_count, openmindedness_distr::Distributions.ContinuousUnivariateDistribution, stubbornness_distr::Distributions.ContinuousUnivariateDistribution)
   opinion = kendall_encoding(vote, can_count)
   openmindedness = rand(openmindedness_distr)
   stubbornness = rand(stubbornness_distr)

   return Kendall_voter(ID, vote, opinion, openmindedness, stubbornness)
end

function init_voters(election, can_count, voter_config::Kendall_voter_init_config)
   openmindedness_distr = Distributions.Truncated(voter_config.openmindedness_distr, 0.0, 1.0)
   stubbornness_distr = Distributions.Truncated(voter_config.stubbornness_distr, 0.0, 1.0)

   voters = Vector{Kendall_voter}(undef, length(election))
   for (i, vote) in enumerate(election)
       voters[i] = Kendall_voter(i, vote, can_count, openmindedness_distr, stubbornness_distr)
   end

   return voters
end

function get_vote(voter::Kendall_voter) :: Vector{Vector{Int}}
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

function step!(self::Kendall_voter, voters, graph, can_count, voter_diff_config::Kendall_voter_diff_config)
   neighbors_ = neighbors(graph, self.ID)
   if length(neighbors_) == 0
      return
   end

   neighbor_id = neighbors_[rand(1:end)]
   neighbor = voters[neighbor_id]

   if rand() <= voter_diff_config.attract_proba
      voters[self.ID] = attract_flip(self, neighbor, can_count)
      voters[neighbor.ID] = attract_flip(neighbor, self, can_count)
   else
      #repel_flip!(self, neighbor)
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
function attract_flip(self, neighbor, can_count, log_lvl=0)
   if get_distance(self, neighbor) == 0.0
      return self
   end

   if rand() > self.stubborness
      actions = get_feasible_actions(self, neighbor, can_count)
      if log_lvl > 1
         display("Feasible actions: ", actions)
      end

      if length(actions) == 0
         println(self.vote)
         println(neighbor.vote)
         error("No feasible actions found, but non-zero distance")
         return
      end
   
      action = actions[rand(1:length(actions))]
      if log_lvl > 1
         println("Action:", action)
      end
      self = apply_action(action, self, can_count)
   end

   return self
end

function delete_can!(vote, bucket_idx, can)
   for i in 1:length(vote[bucket_idx])
      if vote[bucket_idx][i] == can 
         deleteat!(vote[bucket_idx], i)
         break
      end
   end
end

function apply_action(action, voter, can_count)
   can, bucket_idx, _, type = action
   bucket = voter.vote[bucket_idx]
   new_vote = deepcopy(voter.vote)
   
   if type == :unbucket_left
      new_opinion = unbucket_left(can, voter.opinion, bucket, can_count)
      
      delete_can!(new_vote, bucket_idx, can)
      insert!(new_vote, bucket_idx, [can])

   elseif type == :unbucket_right
      new_opinion = unbucket_right(can, voter.opinion, bucket, can_count)
      
      delete_can!(new_vote, bucket_idx, can)
      insert!(new_vote, bucket_idx + 1, [can])

   elseif type == :rebucket_left
      new_opinion = rebucket_left(can, voter.opinion, voter.vote[bucket_idx - 1], bucket, can_count)
      push!(new_vote[bucket_idx - 1], can)
      
      if length(new_vote[bucket_idx]) == 1
         deleteat!(new_vote, bucket_idx)
      else
         delete_can!(new_vote, bucket_idx, can)
      end
      
   else
      new_opinion = rebucket_right(can, voter.opinion, bucket, voter.vote[bucket_idx + 1], can_count)
      push!(new_vote[bucket_idx + 1], can)

      if length(new_vote[bucket_idx]) == 1
         deleteat!(new_vote, bucket_idx)
      else
         delete_can!(new_vote, bucket_idx, can)
      end
   end

   return Kendall_voter(voter.ID, new_vote, new_opinion, voter.openmindedness, voter.stubborness)
end

function get_feasible_actions(u::Kendall_voter, v::Kendall_voter, can_count)
   # vector of new votes and opinions for voter u if we choose specific action 
   feasible_actions = Vector{Tuple{Int64, Int64, Float64, Symbol}}()
   
   # check first bucket possible unbuckets
   if length(u.vote[1]) > 1
      append!(feasible_actions, unbucket(u.opinion, v.opinion, u.vote, 1, can_count))
   end

   for i in 2:length(u.vote)
      # check for possible unbuckets
      if length(u.vote[i]) > 1
         append!(feasible_actions, unbucket(u.opinion, v.opinion, u.vote, i, can_count))
      end

      # check for possible rebuckets
      append!(feasible_actions, rebucket(u.opinion, v.opinion, u.vote, i, can_count))
   end

   return feasible_actions
end

function unbucket(u_opinion, v_opinion, u_vote, bucket_idx, can_count)
   feasible_actions = Vector{Tuple{Int64, Int64, Float64, Symbol}}()
   bucket = u_vote[bucket_idx]
   d_uv = get_distance(u_opinion, v_opinion)

   for can in bucket
      new_u_opinion = unbucket_right(can, u_opinion, bucket, can_count)
      change = get_distance(new_u_opinion, v_opinion) - d_uv

      if change < 0.0 && abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, bucket_idx, change, :unbucket_right))
      end

      new_u_opinion = unbucket_left(can, u_opinion, bucket, can_count)
      change = get_distance(new_u_opinion, v_opinion) - d_uv

      if change < 0.0 && abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, bucket_idx, change, :unbucket_left))
      end
   end

   return feasible_actions
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
   feasible_actions = Vector{Tuple{Int64, Int64, Float64, Symbol}}()
   l_bucket = u_vote[r_bucket_idx - 1]
   r_bucket = u_vote[r_bucket_idx]
   d_uv = get_distance(u_opinion, v_opinion)
   
   for can in l_bucket
      new_u_opinion = rebucket_right(can, u_opinion, l_bucket, r_bucket, can_count)
      change = get_distance(new_u_opinion, v_opinion) - d_uv

      if change < 0.0 && abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, r_bucket_idx - 1, change, :rebucket_right))
      end
   end

   for can in r_bucket
      new_u_opinion = rebucket_left(can, u_opinion, l_bucket, r_bucket, can_count)
      change = get_distance(new_u_opinion, v_opinion) - d_uv
      
      if change < 0.0 && abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, r_bucket_idx, change, :rebucket_left))
      end
   end

   return feasible_actions
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
Gets all inverted candidate pairs indexes based on opinion difference.
"""
function get_all_actions(voter_1, voter_2)
   actions = Vector{Int64}()
   diff = voter_1.opinion - voter_2.opinion

   for i in 1:length(diff)
      if val != 0
         push!(actions, i)
      end
   end

   return actions
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

function test_random_KT(n, can_count, log_lvl=0)
   #fst = Kendall_voter(69, vote_1, kendall_encoding(vote_1, can_count), 0.0, 0.0)
   #snd = Kendall_voter(69, vote_2, kendall_encoding(vote_2, can_count), 0.0, 0.0)
   voters = Vector{Kendall_voter}(undef, n)
   for i in 1:n
      vote = get_random_vote(can_count)
      voters[i] = Kendall_voter(69, vote, kendall_encoding(vote, can_count), 0.0, 0.0)
   end

   for i in 1:n
      for j in 1:n
         if i == j 
            break
         end

         test_pair_KT(voters[i], voters[j], can_count, log_lvl=log_lvl)
      end
   end
end

function test_pair_KT(fst, snd, can_count, log_lvl=0)
   fst = deepcopy(fst)
   snd = deepcopy(snd)

   dist = get_distance(fst, snd)
   println("init d_uv:", dist)
   k = 0
   while dist != 0.0 
      if log_lvl > 1
         println(fst)
         println(snd)
      end

      new_fst = attract_flip(fst, snd, can_count, log_lvl=log_lvl)

      if k == can_count*can_count
         error("Failed to converge in time")
         break
      end
      k += 1
      new_dist = get_distance(new_fst, snd)

      
      if dist - new_dist != get_distance(fst, new_fst)
         println("u:    ", fst.vote)
         println("a(u): ", new_fst.vote)
         println("v:    ", snd.vote)
         println("u, a(u): ", get_distance(fst, new_fst))
         println("a(u), v: ", dist - new_dist, " (change)")
         error("Chosen sub-optimal action")
         break
      end

      fst = new_fst
      dist = new_dist
   end
   println("converged after:", k)
end