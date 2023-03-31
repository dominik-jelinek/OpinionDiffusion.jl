struct Candidate
   ID::Int64
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
   if split(input_filename, '.')[end] == "toc"
      election = parse_votes(lines, length(candidates))
   elseif split(input_filename, '.')[end] == "soi"
      election = parse_votes2(lines, length(candidates))
   else
      throw(ArgumentError("Unsupported format of input data. Supported: [toc, soi]"))
   end

   return parties, candidates, election
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
         candidates[i] = Candidate(i, line[1:end-length(party)-2], length(parties))
      else
         candidates[i] = Candidate(i, line[1:end-length(party)-2], findfirst(x->x==party, parties)) 
      end
   end

   return parties, candidates
end

function parse_votes(lines, can_count) ::Vector{Vote} 
   voters_count = parse(Int, split(lines[2+can_count], ",")[1])
   voters_uniq = parse(Int, split(lines[2+can_count], ",")[3])

   election = Vector{Vote}(undef, voters_count)
   counter = 1
   for i in 1:voters_uniq
      line = lines[i + can_count + 2]
      voter_str = split(line, "{")
      vote = Vote()
      tokenized_vote = length(voter_str) == 2 ? split(chop(voter_str[1]), ",") : split(voter_str[1], ",")
      #first column contains number of matching votes
      matching_votes = parse(Int, tokenized_vote[1])

      #bucket for each known prefference
      for j in 2:length(tokenized_vote)
         push!(vote, Bucket(parse(Int64, tokenized_vote[j])))
      end
      
      #the rest goes into one bucket as they are not distinguishable
      if length(voter_str) == 2
         no_pref = split(chop(voter_str[2]), ",")
         bucket = Bucket()
         for j in eachindex(no_pref)
            push!(bucket, parse(Int64, no_pref[j]))
         end
         push!(vote, bucket)
      end
      
      #copy parsed vote into election based on how many matching votes there were
      for j in 1:matching_votes
         election[counter] = deepcopy(vote)
         counter += 1
      end
   end

   return election
end

#soi format outdated
function parse_votes2(lines, can_count)
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