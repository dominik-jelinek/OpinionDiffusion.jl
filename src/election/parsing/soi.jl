"""
	parse_data(path_data::String, ::Val{:soi})::Election

Parses input file and initializes DB from it

# Arguments
- `path_data::String`: Path to input file

# Returns
- `election::Election`: The parsed election.

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
function parse_data(path_data::String, ::Val{:soi})::Election
	f = open(path_data, "r")
	lines = readlines(f)
	close(f)

	parties, candidates = parse_candidates_soi(lines)
	votes = parse_votes_soi(lines, length(candidates))

	return Election(parties, candidates, votes)
end

"""
	parse_candidates_soi(lines::Vector{String})::Tuple{Vector{String}, Vector{Candidate}}

Parses candidates from input file

# Arguments
- `lines::Vector{String}`: Lines of input file

# Returns
- `parties::Vector{String}`: Vector of parties
- `candidates::Vector{Candidate}`: Vector of candidates
"""
function parse_candidates_soi(lines)
	can_count = parse(Int, lines[1])

	candidates = Vector{Candidate}(undef, can_count)
	parties = Vector{String}()

	for i in 1:can_count
		line = split(lines[i+1], ",")[2]
		party = split(line)[end]

		if party == "Non-P" || !(party in parties)
			push!(parties, party)
			candidates[i] = Candidate(i, line[1:end-length(party)-2], length(parties))
		else
			candidates[i] = Candidate(i, line[1:end-length(party)-2], findfirst(x -> x == party, parties))
		end
	end

	return parties, candidates
end

"""
	parse_votes_soi(lines::Vector{String}, can_count::Int)::Matrix{Int}

Parses votes from input file

# Arguments
- `lines::Vector{String}`: Lines of input file
- `can_count::Int`: Number of candidates

# Returns
- `database::Matrix{Int}`: Matrix of votes
"""
function parse_votes_soi(lines, can_count)
	voters_count = parse(Int, split(lines[2+can_count], ",")[1])
	voters_uniq = parse(Int, split(lines[2+can_count], ",")[3])

	database = zeros(Int, can_count, voters_count)
	counter = 1
	for i in 1:voters_uniq
		line = lines[i+can_count+2]
		voteStr = split(line, ",")
		buffer = zeros(Int, can_count)

		#first column contains number of matching votes
		matchingVotes = parse(Int, voteStr[1])

		for j in 2:length(voteStr)
			buffer[j-1] = parse(Int, voteStr[j])
		end

		for j in 1:matchingVotes
			database[1:can_count, counter] .= buffer
			counter += 1
		end
	end

	return database
end
