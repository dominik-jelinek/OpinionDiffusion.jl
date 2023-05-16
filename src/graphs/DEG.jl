@kwdef struct DEG_graph_config <: Abstract_graph_init_config
    exp::Float64
	scale::Float64
    max_degree::Int64
    target_cc::Float64
    homophily::Float64
    #openmindedness_distr::Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
end

function init_graph(voters, graph_init_config::DEG_graph_config; rng=Random.GLOBAL_RNG)
	pareto = Distributions.truncated(Distributions.Pareto(graph_init_config.exp, graph_init_config.scale); upper=graph_init_config.max_degree)
	target_deg_distr = Int.(round.(rand(rng, pareto, length(voters))))

    return get_DEG(
        voters,
        target_deg_distr,
        graph_init_config.target_cc,
        homophily=graph_init_config.homophily,
        rng=rng
    )
end

function get_DEG(voters, targed_deg_distr, target_cc; rng=Random.GLOBAL_RNG, homophily=0.0)
    n = length(voters)
    social_network = SimpleGraph(n)

    M = floor(sum(targed_deg_distr) / 2)
    rds = Random.shuffle(rng, targed_deg_distr)
    T = floor(target_cc * sum([choose2(rd) for rd in rds]) / 3)

    #println("M: ", M, " T: ", T)
    limit = M * 10
    i = 0
    distances = get_distance(voters)
    distances = (1/2) .^ distances

    while T > 0
        if limit <= i
            break
        end
        i += 1

        nonzero = findall(x -> x > 0, rds)
        rds_nonzero = rds[nonzero]
        u = StatsBase.sample(rng, 1:length(rds_nonzero), StatsBase.Weights(rds_nonzero))

        distances_u = distances[u, nonzero]
        probs = ((1.0 - homophily) .* rds_nonzero ./ sum(rds_nonzero)) .+ homophily .* distances_u ./ sum(distances_u)
        probs[u] = 0.0
        v = StatsBase.sample(rng, 1:n, StatsBase.Weights(probs))

        distances_v = distances[v, nonzero]
        probs = (1.0 - homophily) .* rds_nonzero ./ sum(rds_nonzero) .+ homophily .* (distances_u ./ sum(distances_u) .+ distances_v ./ sum(distances_v)) ./ 2
        probs[u] = 0.0
        probs[v] = 0.0
        w = StatsBase.sample(rng, 1:n, StatsBase.Weights(probs))

        u, v, w = nonzero[u], nonzero[v], nonzero[w]

        # test if we hae sufficient remaining degrees for creating a triangle in between vertices u, v, w
        if rds[u] == 1 && !has_edge(social_network, u, v) && !has_edge(social_network, u, w)
            continue
        end

        if rds[v] == 1 && !has_edge(social_network, v, u) && !has_edge(social_network, v, w)
            continue
        end

        if rds[w] == 1 && !has_edge(social_network, w, u) && !has_edge(social_network, w, v)
            continue
        end

        for (v_1, v_2) in zip([u, v, w], [v, w, u])
            if add_edge!(social_network, v_1, v_2)
                M -= 1
                rds[v_1] -= 1
                rds[v_2] -= 1
                T -= length(common_neighbors(social_network, v_1, v_2))
                i = 0
                limit = M * 10
            end
        end

        # we decrease T even when no new triangles were created as is written in the paper
        #T -= 1
    end
    #println("M: ", M, " T: ", T)
    #println(global_clustering_coefficient(social_network))
    
    i = 0
    while M > 0
        if limit <= i
            break
        end
        i += 1

        nonzero = findall(x -> x > 0, rds)
        rds_nonzero = rds[nonzero]
        u = StatsBase.sample(rng, 1:length(rds_nonzero), StatsBase.Weights(rds_nonzero))

        distances_u = distances[u, nonzero]
        probs = (1.0 - homophily) .* rds_nonzero ./ sum(rds_nonzero) .+ homophily .* distances_u ./ sum(distances_u)
        probs[u] = 0.0
        v = StatsBase.sample(rng, 1:n, StatsBase.Weights(probs))

        u, v = nonzero[u], nonzero[v]

        if add_edge!(social_network, u, v)
            rds[u] -= 1
            rds[v] -= 1
            M -= 1
            i = 0
            limit = M * 10
        end
    end

    return social_network
end