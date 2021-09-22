struct Candidate
   name::String
   party::Int64
end

"""
Parses input file and initializes DB from it
## Input data:
```
9
1,Robert Bonnie G.P. 
2,Joan Burton Lab 
...
9,Sheila Terry F.G. 
29988,29988,10335
621,5,3,7
555,5,3
452,4
...
1,9,2,7,3,5,4,6,8,1
1,4,9,2,1,6,3
```
""" 
function parse_data(input_filename::String)
   f = open("data/$input_filename","r")
   lines = readlines(f)
   close(f)

   parties, candidates = parse_candidates(lines)
   database = parse_votes(lines, length(candidates))

   return parties, candidates, database
end
"""
Parses data
"""
function parse_data2(input_filename::String)
   f = open("data/$input_filename","r")
   lines = readlines(f)
   close(f)

   parties, candidates = parse_candidates(lines)
   database = parse_votes2(lines, length(candidates))

   return parties, candidates, database
end

function parse_candidates(lines)
   can_count = parse(Int,lines[1])

   candidates = Vector{Candidate}(undef, can_count)
   parties = Vector{String}()

   for i in 1:can_count
      line = split(lines[i+1], ",")[2]
      party = split(line)[end]

      if party == "Non-P" || !(party in parties)
         push!(parties, party)
         candidates[i] = Candidate(line[1:end-length(party)-2], length(parties))
      else
         candidates[i] = Candidate(line[1:end-length(party)-2], findfirst(x->x==party, parties)) 
      end
   end

   return parties, candidates
end

function parse_votes2(lines, can_count)
   voters_count = parse(Int, split(lines[2+can_count], ",")[1])
   voters_uniq = parse(Int, split(lines[2+can_count], ",")[3])

   election = Vector{Vector{Vector{Int64}}}(undef, voters_count)
   counter = 1
   for i in 1:voters_uniq
      line = lines[i + can_count + 2]
      voter_str = split(line, "{")
      bucket_vote = Vector{Vector{Int64}}()
      vote = length(voter_str) == 2 ? split(chop(voter_str[1]), ",") : split(voter_str[1], ",")

      #first column contains number of matching votes
      matching_votes = parse(Int, vote[1])

      #bucket for each known prefference
      for j in 2:length(vote)
         push!(bucket_vote, [parse(Int64, vote[j])])
      end
      
      #the rest goes into one bucket as they are not distinguishable
      if length(voter_str) == 2
         no_pref = split(chop(voter_str[2]), ",")
         bucket = Vector{Int64}()
         for j in 1:length(no_pref)
            push!(bucket, parse(Int64, no_pref[j]))
         end
         push!(bucket_vote, bucket)
      end

      for j in 1:matching_votes
         election[counter] = deepcopy(bucket_vote)
         counter += 1
      end
   end

   return election
end

function parse_votes(lines, can_count)
   voters_count = parse(Int, split(lines[2+can_count], ",")[1])
   voters_uniq = parse(Int, split(lines[2+can_count], ",")[3])

   database = zeros(Int, can_count, voters_count)
   counter = 1
   for i in 1:voters_uniq
      line = lines[i + can_count + 2]
      voteStr = split(line,",")
      buffer = zeros(Int, can_count)

      #first column contains number of matching votes
      matchingVotes = parse(Int, voteStr[1])

      for j in 2:length(voteStr)
         buffer[j - 1] = parse(Int, voteStr[j])
      end

      for j in 1:matchingVotes
         database[1:can_count, counter] .= buffer
         counter += 1
      end
   end

   return database
end

function expr2fun(ex)
   vars = getvars(ex)
   return eval(:($Expr(:meta, :inline); $vars -> $ex) )
end

getvars(ex) = Expr(:tuple, _getvars(ex)...)
_getvars(ex::Symbol) = [ex]
function _getvars(ex::Expr)
   vars = Symbol[]
   if ex.head == :call
       for arg in ex.args[2:end]
           append!(vars, _getvars(arg) )
       end
   end
   return unique(vars)
end
_getvars(ex) = Symbol[]

function parse_function(function_config::Dict{String})
   if function_config["type"] == "exp"
      func = x -> (function_config["base"])^(x - (haskey(function_config, "offset") ? function_config["offset"] : 0.0))

   elseif function_config["type"] == "id"
      func = x -> x

   elseif function_config["type"] == "power"
      func = x -> x^function_config["power"]

   elseif function_config["type"] == "constant"
      func = x -> function_config["val"]

   elseif function_config["type"] == "inverse"
      func = x -> 1/x

   elseif function_config["type"] == "compound"
      func = x -> parse_function["f"](parse_function["g"](x))

   elseif function_config["type"] == "expr"
      func = Base.invokelatest(Meta.parse(function_config["lambda"]))
   else
      error("Unknown function type [exp | id | power | expr]")
   end

   return func
end

function parse_metric(metric::String)
   if metric == "L1"
      distMetric = Cityblock()
   elseif metric == "L2"
      distMetric = Euclidean()
   else
      error("Unknown metric, supported: [L1 | L2]")
   end
   return distMetric
end

function parse_distribution(distribution_config::Dict{String, Any})
   if distribution_config["type"] == "normal"
      @assert haskey(distribution_config, "mean")
      @assert haskey(distribution_config, "variance")
      distr = Normal(distribution_config["mean"], distribution_config["variance"])

   elseif distribution_config["type"] == "uniform"
      @assert haskey(distribution_config, "lower_bound")
      @assert haskey(distribution_config, "upper_bound")
      distr = Uniform(distribution_config["lower_bound"], distribution_config["upper_bound"])

   elseif distribution_config["type"] == "exponential"
      @assert haskey(distribution_config, "scale")
      distr = Exponential(distribution_config["scale"])
   
   else 
      error("Unknown distribution, supported: [normal | uniform | exponential]")
   end

   return distr
end

#_________________________________________

function compressDB(database)
   compressedDBpref = Array{Int,1}[]
   compressedDBgrp = Array[]

   for i in 1:length(database)
      used = false
      for j in 1:length(compressedDBpref)
         if database[i] == compressedDBpref[j]
            push!(compressedDBgrp[j], i)
            used = true
         end
      end
      if !used
         push!(compressedDBpref, database[i])
         push!(compressedDBgrp, [i])
      end
   end
   return compressedDBpref, compressedDBgrp
end

function compressGroup(database, group)
   voters = Int[]
   compressedGr = Array{Int,1}[]
   for i in 1:length(group)
      used = false
      for j in 1:length(compressedGr)
         if database[group[i]] == compressedGr[j]
            voters[j] += 1
            used = true
         end
      end
      if !used
         push!(compressedGr, database[group[i]])
         push!(voters, 1)
      end
   end
   return voters, compressedGr
end