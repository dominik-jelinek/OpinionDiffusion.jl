@kwdef struct BA_graph_config <: Abstract_graph_config
	rng_seed::UInt32
	average_degree::Int64
	homophily::Float64
end
name(type::Type{BA_graph_config}) = "Barabasi-Albert"

function init_graph(voters::Vector{T}, graph_config::BA_graph_config) where {T<:Abstract_voter}
	rng = Random.MersenneTwister(graph_config.rng_seed)
	return barabasi_albert_graph(voters, graph_config.average_degree; homophily=graph_config.homophily, rng=rng)
end

function barabasi_albert_graph(voters::Vector{T}, average_degree::Int64; homophily=0.0, rng=Random.GLOBAL_RNG) where {T<:Abstract_voter}
	m = ceil(Int64, average_degree / 2)
	if m > length(voters)
		throw(ArgumentError("Argument average_degree for Barabasi-Albert graph creation is higher than number of voters."))
	end
	n = length(voters)
	social_network = SimpleGraph(n)

	rand_perm = Random.shuffle(rng, 1:n)
	voters_perm = voters[rand_perm]

	# add first node that is connected to all other nodes
	@inbounds for i in 1:m
		add_edge!(social_network, rand_perm[m+1], rand_perm[i])
		#set_prop!(social_network, rand_perm[m+1], rand_perm[i], :weight, get_distance(voters_perm[m+1], voters_perm[i]))
	end

	degrees = zeros(Float64, n)
	for i in 1:m
		degrees[i] = 1.0
	end
	degrees[m+1] += m
	degree_sum = m

	# add the rest of nodes and generate edges based on opinion similarity and popularity of other votes
	probs = zeros(Float64, n)
	for i in m+2:length(voters_perm)
		self = voters_perm[i]

		distances = (1 / 2) .^ get_distance(self, voters_perm[1:i-1])
		#calculate distribution of p robabilities for each previously added vertex
		dist_sum = sum(distances)
		@inbounds for j in eachindex(distances)
			probs[j] = (1.0 - homophily) * degrees[j] / degree_sum + homophily * distances[j] / dist_sum
		end
		edge_ends = StatsBase.sample(rng, 1:n, StatsBase.Weights(probs), m, replace=false)


		#add edges
		@inbounds for edge_end in edge_ends
			add_edge!(social_network, rand_perm[i], rand_perm[edge_end])
			#set_prop!(social_network, rand_perm[i], rand_perm[edge_end], :weight, 1.0)#distances[edge_end])
			degrees[edge_end] += 1.0
		end

		degrees[i] += m
		degree_sum += 2 * m
	end

	return social_network
end