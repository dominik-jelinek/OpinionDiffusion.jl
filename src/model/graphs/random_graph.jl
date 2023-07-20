@kwdef struct Random_graph_config <: Abstract_graph_config
	rng_seed::UInt32
	average_degree::Int64
end
name(type::Type{Random_graph_config}) = "Random"

"""
	init_graph(voters::Vector{T}, graph_config::Random_graph_config) where {T<:Abstract_voter}

Initializes the given voters with the given graph_config.

# Arguments
- `voters::Vector{T}`: The voters to initialize.
- `graph_config::Random_graph_config`: The config to initialize the voters with.

# Returns
- `voters::Vector{T}`: The initialized voters.
"""
function init_graph(voters::Vector{T}, graph_config::Random_graph_config) where {T<:Abstract_voter}
	rng = MersenneTwister(graph_config.rng_seed)
	#return random_regular_graph(length(voters), graph_config.average_degree; rng=rng)

	return random_graph(voters, graph_config.average_degree; rng=rng)
end

"""
	random_graph(voters::Vector{T}, average_degree::Int64; rng=Random.GLOBAL_RNG) where {T<:Abstract_voter}

Returns a social network with the given average degree.

# Arguments
- `voters::Vector{T}`: The voters to initialize.
- `average_degree::Int64`: The average degree.
- `rng::Random.MersenneTwister`: The random number generator to use.

# Returns
- `social_network::SimpleGraph`: The social network.
"""
function random_graph(voters::Vector{T}, average_degree::Int64; rng=Random.GLOBAL_RNG) where {T<:Abstract_voter}
	n = length(voters)
	social_network = SimpleGraph(n)

	edges = ceil(average_degree * n / 2)
	for i in 1:edges
		u = rand(rng, 1:n)
		v = rand(rng, 1:n)
		while u == v || has_edge(social_network, u, v)
			u = rand(rng, 1:n)
			v = rand(rng, 1:n)
		end
		add_edge!(social_network, u, v)
	end

	return social_network
end
