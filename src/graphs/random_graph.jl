@kwdef struct random_graph_config <: Abstract_graph_init_config
    rng_seed::UInt32
    average_degree::Float64
end

function init_graph(voters::Vector{T}, graph_init_config::random_graph_config) where {T<:Abstract_voter}
    rng = MersenneTwister(graph_init_config.rng_seed)
    return random_regular_graph(length(voters), average_degree; rng=rng)
    #return random_graph(voters, graph_init_config.average_degree; rng=rng)
end

function random_graph(voters::Vector{T}, average_degree::Float64; rng=Random.GLOBAL_RNG) where {T<:Abstract_voter}
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