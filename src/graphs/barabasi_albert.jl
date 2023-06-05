@kwdef struct BA_graph_config <: Abstract_graph_init_config
    rng::Random.MersenneTwister
    m::Integer
    homophily::Real
end

function BA_graph_config(m::Integer, homophily::Real)
    rng = Random.MersenneTwister(rand(UInt32))
    return BA_graph_config(rng, m, homophily)
end

function init_graph(voters, graph_init_config::BA_graph_config;)
    return barabasi_albert_graph(voters, graph_init_config.m; homophily=graph_init_config.homophily, rng=graph_init_config.rng)
end

function barabasi_albert_graph(voters::Vector{T}, m::Integer; homophily=0.0, rng=Random.GLOBAL_RNG) where {T<:Abstract_voter}
    if m > length(voters)
        throw(ArgumentError("Argument m for Barabasi-Albert graph creation is higher than number of voters."))
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