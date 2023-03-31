
struct Kendall_voter <: Abstract_voter
   ID::Int64
   opinion::Vector{Float64}

   vote::Vote
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

function init_voters(election, can_count, voter_config::Kendall_voter_init_config; rng=Random.GLOBAL_RNG)
   openmindedness_distr = Distributions.Truncated(voter_config.openmindedness_distr, 0.0, 1.0)
   stubbornness_distr = Distributions.Truncated(voter_config.stubbornness_distr, 0.0, 1.0)

   voters = Vector{Kendall_voter}(undef, length(election))
   for (i, vote) in enumerate(election)
      opinion = kendall_encoding(vote, can_count)
      openmindedness = rand(rng, openmindedness_distr)
      stubbornness = 0.5#rand(rng, stubbornness_distr)
      voters[i] = Kendall_voter(i, opinion, vote, openmindedness, stubbornness)
   end

   return voters
end

function get_vote(voter::Kendall_voter) :: Vote
   return voter.vote
end

function get_pos(voter::Kendall_voter, can)
   pos = 0
   for bucket in get_vote(voter)
      if can in bucket 
         return pos + (length(bucket) + 1)/ 2
      end

      pos += length(bucket)
   end

   return pos
end

"""
Encodes vote into space of dimension can_ount choose 2 
"""
function kendall_encoding(vote::Vote, can_count)
   inv_vote = invert_vote(vote, can_count)
   n = choose2(can_count)

   opinion = Vector{Float64}(undef, n)
   counter = 1
   for can_1 in 1:can_count-1
      for can_2 in can_1+1:can_count
         opinion[counter] = get_penalty(inv_vote[can_1], inv_vote[can_2], n)
         counter += 1
      end
   end

   return opinion
end

function step!(self::Kendall_voter, model, voter_diff_config::Kendall_voter_diff_config; rng=Random.GLOBAL_RNG)
   voters = get_voters(model)
   social_network = get_social_network(model)
   neighbors_ = neighbors(social_network, self.ID)
   can_count = length(model.candidates) 
   if length(neighbors_) == 0
      return []
   end

   neighbor_id = neighbors_[rand(rng, 1:length(neighbors_))]
   neighbor = voters[neighbor_id]

   if rand(rng) <= voter_diff_config.attract_proba
      method = "attract"
      voters[self.ID] = attract(self, neighbor, can_count)
      voters[neighbor.ID] = attract(neighbor, self, can_count)
   else
      method = "repel"
      voters[self.ID] = repel(self, neighbor, can_count)
      voters[neighbor.ID] = repel(neighbor, self, can_count)
   end

   return [Action(method, (neighbor.ID, self.ID), self, voters[self.ID]), Action("repel", (self.ID, neighbor.ID), neighbor, voters[neighbor.ID])]
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
function attract(self, neighbor, can_count; rng=Random.GLOBAL_RNG, log_lvl=0)
   if get_distance(self, neighbor) == 0.0
      return self
   end

   if rand(rng) > self.stubborness
      #actions = get_feasible_actions(self, neighbor, can_count)
      actions = get_distance_preserving_actions_dec(self, neighbor, can_count)

      if log_lvl > 2
         println("Feasible actions: ", actions)
      end

      if length(actions) == 0
         println(self.vote)
         println(neighbor.vote)
         error("No feasible actions found, but non-zero distance")
         return
      end
      
      # filter restack operation whenever unstack is available, resulting in smaller steps
      min_change = maximum([x[3] for x in actions])
      actions = [action for action in actions if action[3] == min_change]
      action = actions[rand(rng, 1:length(actions))]

      if log_lvl > 1
         println("Action:", action)
      end

      #self = apply_action(action, self, can_count)
      if rand(rng) < 0.5 / abs(action[3])
         self = apply_action2(action, self, can_count)
      end
   end

   return self
end

function repel(self, neighbor, can_count; log_lvl=0, rng=Random.GLOBAL_RNG)
   # we are assuming that opinions are normalized
   max_distance = choose2(can_count)
   max_possible = max_distance
   for bucket in neighbor.vote
      if length(bucket) > 1
         max_possible -= choose2(length(bucket)) * 0.5
      end
   end

   if isapprox(get_distance(self, neighbor) * max_distance, max_possible)
      return self
   end

   if rand(rng) > self.stubborness
      #actions = get_feasible_actions(self, neighbor, can_count)
      actions = get_distance_preserving_actions_inc(self, neighbor, can_count)

      if log_lvl > 2
         println("DP inc: ", actions)
      end

      if length(actions) == 0
         println(to_string(self.vote))
         println(to_string(neighbor.vote))
         println("d_uv: ", get_distance(self, neighbor))
         println("max:  ", max_possible/max_distance)
         error("No DP actions found, but not maximal distance")
         return
      end
      
      # pick the action with the smallest change in KT distance
      min_change = minimum([x[3] for x in actions])
      actions = [action for action in actions if action[3] == min_change]
      action = actions[rand(rng, 1:length(actions))]

      if log_lvl > 1
         println("Action:", action)
      end
      #self = apply_action(action, self, can_count)

      # if the change of KT distance for smallest action is 2 then there is 50% chance that the acttion will be successful
      if rand(rng) < 0.5 / action[3]
         self = apply_action2(action, self, can_count)
      end
   end

   return self
end

function delete_can!(vote, bucket_idx, can)
   for i in eachindex(vote[bucket_idx])
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

function delete_cans!(vote::Vector{Vector{Int64}}, bucket_idx, cans)
   vote[bucket_idx] = [can for can in vote[bucket_idx] if can ∉ cans]
end

function delete_cans!(vote::Vector{Set{Int64}}, bucket_idx, cans)
   for can in cans
      delete!(vote[bucket_idx], can)
   end
end

function apply_action2(action, voter, can_count)
   cans, bucket_idx, _, type = action
   new_vote = deepcopy(voter.vote)

   if type == :unstack_left
      delete_cans!(new_vote, bucket_idx, cans)
      insert!(new_vote, bucket_idx, Bucket(cans))

   elseif type == :unstack_right      
      delete_cans!(new_vote, bucket_idx, cans)
      insert!(new_vote, bucket_idx + 1, Bucket(cans))

   elseif type == :restack_left
      push!(new_vote[bucket_idx - 1], cans...)
      
      if length(new_vote[bucket_idx]) == length(cans)
         deleteat!(new_vote, bucket_idx)
      else
         delete_cans!(new_vote, bucket_idx, cans)
      end
   else
      push!(new_vote[bucket_idx + 1], cans...)

      if length(new_vote[bucket_idx]) == length(cans)
         deleteat!(new_vote, bucket_idx)
      else
         delete_cans!(new_vote, bucket_idx, cans)
      end
   end

   new_opinion = kendall_encoding(new_vote, can_count)

   return Kendall_voter(voter.ID, new_opinion, new_vote, voter.openmindedness, voter.stubborness)
end

function get_extremes(bucket_set, inverted_vote)
   bucket = collect(bucket_set)
   pos_in_v = inverted_vote[bucket]
   pos_min, pos_max = minimum(pos_in_v), maximum(pos_in_v)

   return (bucket[findall(pos_in_v .== pos_min)], pos_min), (bucket[findall(pos_in_v .== pos_max)], pos_max)
end

function get_distance_preserving_actions_dec(u::Kendall_voter, v::Kendall_voter, can_count)
   # vector of new votes and opinions for voter u if we choose specific action 
   # (cans, bucket_idx, distance_change, action)
   DP_actions = Vector{Tuple{Vector{Int64}, Int64, Float64, Symbol}}()
   inverted = invert_vote(get_vote(v), can_count)

   # check first bucket possible unstack
   (l_mins, l_pos_min), (l_maxs, l_pos_max) = get_extremes(u.vote[1], inverted)
   if l_pos_min != l_pos_max
      push!(DP_actions, (l_mins, 1, -length(l_mins) * (length(u.vote[1]) - length(l_mins)) * 0.5, :unstack_left))
      push!(DP_actions, (l_maxs, 1, -length(l_maxs) * (length(u.vote[1]) - length(l_maxs)) * 0.5, :unstack_right))
   end

   for i in 2:length(u.vote)
      (r_mins, r_pos_min), (r_maxs, r_pos_max) = get_extremes(u.vote[i], inverted)
      # check for possible unstack
      if r_pos_min != r_pos_max
         push!(DP_actions, (r_mins, i, -length(r_mins) * (length(u.vote[i]) - length(r_mins)) * 0.5, :unstack_left))
         push!(DP_actions, (r_maxs, i, -length(r_maxs) * (length(u.vote[i]) - length(r_maxs)) * 0.5, :unstack_right))
      end

      # check for possible restack from the left bucket to the right bucket
      # if there are candidates from the right bucket that have position less than right most candidates from left bucket 
      # then at least those pairs of candidates will increase Kendall-tau distance
      if l_pos_max >= r_pos_max 
         l_change = -length(l_maxs) * (length(u.vote[i - 1]) - length(l_maxs)) * 0.5
         r_change = -length(l_maxs) * length(u.vote[i]) * 0.5
         push!(DP_actions, (l_maxs, i - 1, l_change + r_change, :restack_right))
      end

      # check for possible restack from the right bucket to the left bucket
      if r_pos_min <= l_pos_min
         l_change = -length(r_mins) * length(u.vote[i - 1]) * 0.5
         r_change = -length(r_mins) * (length(u.vote[i]) - length(r_mins)) * 0.5
         push!(DP_actions, (r_mins, i, l_change + r_change, :restack_left))
      end

      (l_mins, l_pos_min), (l_maxs, l_pos_max) = (r_mins, r_pos_min), (r_maxs, r_pos_max)
   end

   return DP_actions
end

function get_distance_preserving_actions_inc(u::Kendall_voter, v::Kendall_voter, can_count)
   # vector of new votes and opinions for voter u if we choose specific action 
   DP_actions = Vector{Tuple{Vector{Int64}, Int64, Float64, Symbol}}()
   inverted = invert_vote(get_vote(v), can_count)

   # check first bucket possible unstack
   (l_mins, l_pos_min), (l_maxs, l_pos_max) = get_extremes(u.vote[1], inverted)
   if length(u.vote[1]) > 1
      for can in l_mins
         push!(DP_actions, ([can], 1, (length(u.vote[1]) - 1) * 0.5, :unstack_right))
      end

      for can in l_maxs
         push!(DP_actions, ([can], 1, (length(u.vote[1]) - 1) * 0.5, :unstack_left))
      end
   end

   for i in 2:length(u.vote)
      (r_mins, r_pos_min), (r_maxs, r_pos_max) = get_extremes(u.vote[i], inverted)
      # check for possible unstack of right bucket
      if length(u.vote[i]) > 1
         for can in r_mins
            push!(DP_actions, ([can], i, (length(u.vote[1]) - 1) * 0.5, :unstack_right))
         end

         for can in r_maxs
            push!(DP_actions, ([can], i, (length(u.vote[1]) - 1) * 0.5, :unstack_left))
         end
      end

      # check for possible restack from the left bucket to the right bucket
      if l_pos_min < r_pos_min
         for can in l_mins
            push!(DP_actions, ([can] , i - 1, (length(u.vote[i - 1]) - 1) + (length(u.vote[i])), :restack_right))
         end
      end

      # check for possible restack from the right bucket to the left bucket
      if r_pos_max > r_pos_max
         for can in r_maxs
            push!(DP_actions, ([can], i, (length(u.vote[i - 1])) + (length(u.vote[i]) - 1), :restack_left))
         end
      end

      (l_mins, l_pos_min), (l_maxs, l_pos_max) = (r_mins, r_pos_min), (r_maxs, r_pos_max)
   end

   return DP_actions
end

function unbucket(u_opinion, v_opinion, u_vote, bucket_idx, can_count)
   feasible_actions = Vector{Tuple{Int64, Int64, Float64, Symbol}}()
   bucket = u_vote[bucket_idx]
   d_uv = get_distance(u_opinion, v_opinion)

   for can in bucket
      new_u_opinion = unbucket_right(can, u_opinion, bucket, can_count)
      change = get_distance(new_u_opinion, v_opinion) - d_uv

      if change < 0.0 #&& abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, bucket_idx, change, :unbucket_right))
      end

      new_u_opinion = unbucket_left(can, u_opinion, bucket, can_count)
      change = get_distance(new_u_opinion, v_opinion) - d_uv

      if change < 0.0 #&& abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, bucket_idx, change, :unbucket_left))
      end
   end

   return feasible_actions
end

function unbucket_right(can, opinion, bucket, can_count)
   max_distance = choose2(can_count)

   new_opinion = deepcopy(opinion)
   for i in eachindex(bucket)
      if bucket[i] != can
         new_opinion[get_index(bucket[i], can, can_count)] = can < bucket[i] ? 1.0 / max_distance : 0.0
      end
   end

   return new_opinion
end

function unbucket_left(can, opinion, bucket, can_count)
   max_distance = choose2(can_count)

   new_opinion = deepcopy(opinion)
   for i in eachindex(bucket)
      if bucket[i] != can
         new_opinion[get_index(bucket[i], can, can_count)] = can < bucket[i] ? 0.0 : 1.0 / max_distance
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

      if change < 0.0 #&& abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, r_bucket_idx - 1, change, :rebucket_right))
      end
   end

   for can in r_bucket
      new_u_opinion = rebucket_left(can, u_opinion, l_bucket, r_bucket, can_count)
      change = get_distance(new_u_opinion, v_opinion) - d_uv
      
      if change < 0.0 #&& abs(change) == get_distance(u_opinion, new_u_opinion)
         push!(feasible_actions, (can, r_bucket_idx, change, :rebucket_left))
      end
   end

   return feasible_actions
end

function rebucket_right(can, opinion, l_bucket, r_bucket, can_count)
   max_distance = choose2(can_count)

   new_opinion = deepcopy(opinion)
   for i in eachindex(l_bucket)
      if l_bucket[i] != can
         new_opinion[get_index(l_bucket[i], can, can_count)] = can < l_bucket[i] ? 1.0 / max_distance : 0.0
      end
   end

   for i in eachindex(r_bucket)
      new_opinion[get_index(r_bucket[i], can, can_count)] = 0.5 / max_distance
   end

   return new_opinion
end

function rebucket_left(can, opinion, l_bucket, r_bucket, can_count)
   max_distance = choose2(can_count)

   new_opinion = deepcopy(opinion)

   for i in eachindex(l_bucket)
      new_opinion[get_index(l_bucket[i], can, can_count)] = 0.5 / max_distance
   end

   for i in eachindex(r_bucket)
      if r_bucket[i] != can
         new_opinion[get_index(r_bucket[i], can, can_count)] = can < r_bucket[i] ? 0.0 : 1.0 / max_distance
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

   for i in eachindex(diff)
      if val != 0
         push!(actions, i)
      end
   end

   return actions
end

function invert_vote(vote, can_count)
   pos = Vector{Int64}(undef, can_count)

   for i in eachindex(vote)
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

function get_penalty(pos_1, pos_2, max_distance = nothing)
   # candidates are indistinguishable
   penalty = 0.5
   
   if pos_1 < pos_2
      # candidates are in order with numerical candidate ordering
      penalty = 0.0
   elseif pos_1 > pos_2
      # candidates are out of order with numerical candidate ordering
      penalty = 1.0
   end
   
   return max_distance === nothing ? penalty : penalty / max_distance
end

function test_pair_KT(fst, snd, can_count; log_lvl=0)
   fst = deepcopy(fst)
   snd = deepcopy(snd)
   
   dist = get_distance(fst, snd)
   init_dist = dist
   if log_lvl > 0
      println("init d_uv:", dist)
   end
   k = 0
   while dist != 0.0 
      if log_lvl > 1
         println(to_string(fst.vote))
         println(to_string(snd.vote))
      end

      new_fst = attract(fst, snd, can_count; log_lvl=log_lvl)

      if k == choose2(can_count)*10
         println(to_string(fst.vote))
         println(to_string(snd.vote))
         error("Failed to converge in time:", k)
         break
      end
      k += 1
      new_dist = get_distance(new_fst, snd)

      #=
      if dist - new_dist != get_distance(fst, new_fst)
         println("u:    ", fst.vote)
         println("a(u): ", new_fst.vote)
         println("v:    ", snd.vote)
         println("u, a(u): ", get_distance(fst, new_fst))
         println("a(u), v: ", dist - new_dist, " (change)")
         error("Chosen sub-optimal action")
         break
      end
      =#
      
      fst = new_fst
      dist = new_dist
   end
   if log_lvl > 0
      println("converged after:", k)
   end
   ratio = k/init_dist
   #println("ratio: ", ratio)
   return ratio
end

function test_random_KT(n, can_count; log_lvl=0)
   #fst = Kendall_voter(69, vote_1, kendall_encoding(vote_1, can_count), 0.0, 0.0)
   #snd = Kendall_voter(69, vote_2, kendall_encoding(vote_2, can_count), 0.0, 0.0)
   voters = Vector{Kendall_voter}(undef, n)
   for i in 1:n
      vote = get_random_vote(can_count)
      voters[i] = Kendall_voter(69, kendall_encoding(vote, can_count), vote, 0.0, 0.0)
   end

   ratios = []
   for i in 1:n
      for j in 1:n
         if i == j 
            break
         end
         dist = get_distance(voters[i], voters[j])
         if dist == 0.0
            break
         end

         push!(ratios, test_pair_KT(voters[i], voters[j], can_count, log_lvl=log_lvl))
      end
   end

   println(Statistics.mean(ratios))
end