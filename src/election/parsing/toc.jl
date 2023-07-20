"""
	parse_data(path_data::String, ::Val{:toc})::Election

Parses input file and initializes DB from it

# Arguments
- `path_data::String`: Path to input file

# Returns
- `election::Election`: Parsed election

## Input data:
```
9
1,Robert Bonnie G.P.
2,Joan Burton Lab
...
9,Sheila Terry F.G.
29988,29988,10230
621,5,3,7,{1,2,4,6,8,9}
555,5,3,{1,2,4,6,7,8,9}
452,4,{1,2,3,5,6,7,8,9}
...
1,5,7,2,9,6,3,4,8,1
1,2,4,7,9,5,3,1,{6,8}
```
"""
function parse_data(path_data::String, ::Val{:toc})::Election
	f = open(path_data, "r")
	lines = readlines(f)
	close(f)

	candidates = parse_candidates_toc(lines)
	votes = parse_votes_toc(lines, length(candidates))

	return Election(candidates, votes)
end

"""
	parse_candidates_toc(lines::Vector{String})::Vector{Candidate}

Parses candidates from input file

# Arguments
- `lines::Vector{String}`: Lines of input file

# Returns
- `candidates::Vector{Candidate}`: Vector of candidates
"""
function parse_candidates_toc(lines)
	can_count = parse(Int, lines[1])

	candidates = Vector{Candidate}(undef, can_count)
	parties = Vector{String}()

	for i in 1:can_count
		line = split(lines[i+1], ",")[2]
		party = split(line)[end]

		if party == "Non-P" || !(party in parties)
			push!(parties, party)
			candidates[i] = Candidate(i, line[1:end-length(party)-2], length(parties), party)
		else
			candidates[i] = Candidate(i, line[1:end-length(party)-2], findfirst(x -> x == party, parties), party)
		end
	end

	return candidates
end

"""
	parse_votes_toc(lines, can_count)::Vector{Vote}

Parses votes from input file

# Arguments
- `lines::Vector{String}`: Lines of input file
- `can_count::Int`: Number of candidates

# Returns
- `votes::Vector{Vote}`: Vector of votes
"""
function parse_votes_toc(lines, can_count)::Vector{Vote}
	voters_count = parse(Int, split(lines[2+can_count], ",")[1])
	voters_uniq = parse(Int, split(lines[2+can_count], ",")[3])

	election = Vector{Vote}(undef, voters_count)
	counter = 1
	for i in 1:voters_uniq
		line = lines[i+can_count+2]
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
