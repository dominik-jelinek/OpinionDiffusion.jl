struct Accumulator
    accumulator::Dict{String, Vector{Any}}
    get_metrics::Function
end

function init_accumulator(get_metrics::Function, model::T) where {T<:Abstract_model}
    accumulator = Dict()

    metrics = get_metrics(model)
    for (key, value) in metrics
        accumulator[key] = [value]
    end

    return Accumulator(accumulator, get_metrics)
end

function add_metrics!(accumulator::Accumulator, model::T) where {T<:Abstract_model}
    metrics = accumulator.get_metrics(model)

    accumulator = accumulator.accumulator
    for (key, value) in metrics
        push!(accumulator[key], value)
    end
end

function accumulated_metrics(accumulator::Accumulator)
    return Dataframe(accumulator)
end

function get_metrics(model)
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
        "avg_edge_dist" => StatsBase.mean(get_edge_distances(g, voters)),
        "clustering_coeff" => Graphs.global_clustering_coefficient(g),
        #"diameter" => Graphs.diameter(g),
        
        "avg_vote_length" => StatsBase.mean([length(vote) for vote in votes]),
        "unique_votes" => length(unique(votes)),
        
        "plurality_scores" => plurality_voting(votes, can_count, true),
        "borda_scores" => borda_voting(votes, can_count, true),
        #"copeland_scores" => copeland_voting(votes, can_count),
        "positions" => get_positions(voters, can_count)
	)
	
	return metrics
end