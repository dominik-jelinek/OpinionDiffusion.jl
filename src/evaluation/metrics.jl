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
        "avg_edge_dist" => OpinionDiffusion.StatsBase.mean(OpinionDiffusion.get_edge_distances(g, voters)),
        "clustering_coeff" => Graphs.global_clustering_coefficient(g),
        #"diameter" => Graphs.diameter(g),
        
        "avg_vote_length" => OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes]),
        "unique_votes" => length(unique(votes)),
        
        "plurality_votings" => plurality_voting(votes, can_count, true),
        "borda_votings" => borda_voting(votes, can_count, true),
        #"copeland_votings" => copeland_voting(votes, can_count),
        "positions" => get_positions(voters, can_count)
	)
	
	return metrics
end

function init_accumulator(metrics::Dict)
    accumulator = Dict()

    for (key, value) in metrics
        accumulator[key] = [value]
    end

    return accumulator
end

function add_metrics!(accumulator, metrics::Dict)
    for (key, value) in metrics
        push!(accumulator[key], value)
    end
end

    
function metrics_vis(metrics, candidates, parties, exp_dir=Nothing)
    degrees = draw_range(metrics["min_degrees"], metrics["avg_degrees"], metrics["max_degrees"], title="Degree range", xlabel="Timestamp", ylabel="Degree", value_label="avg")

    plurality = draw_voting_res(candidates, parties, reduce(hcat, metrics["plurality_votings"])', "Plurality voting")
    borda = draw_voting_res(candidates, parties, reduce(hcat, metrics["borda_votings"])', "Borda voting")
    copeland = draw_voting_res(candidates, parties, reduce(hcat, metrics["copeland_votings"])', "Copeland voting")

    plots = Plots.plot(degrees, plurality, borda, copeland, layout=(2, 2), size=(669, 900))

    if exp_dir != Nothing
        Plots.savefig(plots, "$(exp_dir)/images/metrics.png")
    end

    return plots
end

function draw_range(min, value, max; c=1, label, x=nothing)
    plot = Plots.plot()
    draw_range!(plot, min, value, max; c=c, label=label, x=x)

    return plot
end

function draw_range!(plot, min, value, max; c=1, label, linestyle=:solid, x=nothing)
    if x === nothing
        x = 1:length(value)
    end

    Plots.plot!(plot, x, min, fillrange=max, fillalpha=0.4, c=c, linewidth=0, label=label)
    Plots.plot!(plot, x, value, linewidth=3, label=label, c=c, linestyle=linestyle)
end

function draw_metric(values, title::String; linestyle=:solid, log_idx=nothing)
    plot = Plots.plot()
    draw_metric!(plot, values, title; log_idx=log_idx, linestyle=linestyle)

    return plot
end

function draw_metric!(plot, values, title::String; linestyle=:solid, log_idx=nothing)
    label = log_idx === nothing ? title : title * " " * string(log_idx)
    c = log_idx === nothing ? 1 : log_idx

    draw_range!(plot, [x[2] for x in values], [x[3] for x in values], [x[4] for x in values], label=label, linestyle=linestyle, c=c)
    Plots.plot!(plot, title=title, xlabel="Timestamp", ylabel="Value", yformatter=:plain)

    return plot
end

#___________________________________________________________________
# ENSEMBLE

"""
Gather data from the logs of multiple diffusion experiments and visualize spreads
"""
function gather_metrics(ens_metrics)
    if length(ens_metrics) == 0
        return
    end

    res = Dict()
    for metric in keys(ens_metrics[1])
        # create a matrix out of all the runs for specific metric
        matrix = transpose(hcat([run[metric] for run in ens_metrics]...))

        if matrix[1, 1] isa Number
            # number
            res[metric] = [Statistics.quantile(col, [0.0, 0.25, 0.5, 0.75, 1.0]) for col in eachcol(matrix)]

        else
            # vector
            res[metric] = []
            for col in eachcol(matrix)
                matrix_vect = vcat(col...)
                push!(res[metric], [Statistics.quantile(col, [0.0, 0.25, 0.5, 0.75, 1.0]) for col in eachcol(matrix_vect)])
            end
        end
    end

    return res
end