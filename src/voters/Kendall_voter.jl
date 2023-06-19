
struct Kendall_voter <: Abstract_voter
   ID::Int64

   opinion::Vector{Float64} #pKT
   vote::Vote # BO

   properties::Dict{String,Any}
end

@kwdef struct Kendall_voter_init_config <: Abstract_voter_init_config
   can_count::Int64
end

function init_voters(votes::Vector{Vote}, voter_config::Kendall_voter_init_config)

   voters = Vector{Kendall_voter}(undef, length(votes))
   for (i, vote) in enumerate(votes)
      opinion = kendall_encoding(vote, voter_config.can_count)

      properties = Dict()
      voters[i] = Kendall_voter(i, opinion, vote, properties)
   end

   return voters
end

function get_vote(voter::Kendall_voter)::Vote
   return voter.vote
end

function get_pos(voter::Kendall_voter, can)
   pos = 0
   for bucket in get_vote(voter)
      if can in bucket
         return pos + (length(bucket) + 1) / 2
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