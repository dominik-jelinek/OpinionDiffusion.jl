struct General_model <: Abstract_model
    voters::Vector{Abstract_voter}

    social_network::AbstractGraph

    parties::Vector{String}
    candidates::Vector{Candidate}
end

function eval(model::General_model)
    g = get_social_network(model)
    voters = get_voters(model)
    candidates = get_candidates(model)
    can_count = length(candidates)

    histogram = Graphs.degree_histogram(g)
    keyss = collect(keys(histogram))

    votes = get_votes(voters)

    metrics = Dict(
        "min_degrees" => minimum(keyss),
        "avg_degrees" => Graphs.ne(g) * 2 / Graphs.nv(g),
        "max_degrees" => maximum(keyss),
        "avg_edge_dist" => OpinionDiffusion.StatsBase.mean(OpinionDiffusion.get_edge_distances(g, voters)),
        "clustering_coeff" => Graphs.global_clustering_coefficient(g),
        #"diameter" => Graphs.diameter(g),

        "avg_vote_length" => OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes]),
        "unique_votes" => length(unique(votes)), "plurality_votings" => plurality_voting(votes, can_count, true),
        "borda_votings" => borda_voting(votes, can_count, true),
        #"copeland_votings" => copeland_voting(votes, can_count),
        "positions" => get_positions(voters, can_count)
    )

    return metrics
end

function add_metrics(accumulator, metrics::Dict)
    if accumulator === nothing
        accumulator = Dict()

        for (key, value) in metrics
            accumulator[key] = [value]
        end
        return metrics
    end

    for (key, value) in metrics
        push!(accumulator[key], value)
    end

    return accumulator
end