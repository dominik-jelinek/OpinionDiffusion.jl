function init_metrics(model)
    g = get_social_network(model)
    voters = get_voters(model)
	candidates = get_candidates(model)
	can_count = length(candidates)
	
	histogram = Graphs.degree_histogram(g)
    keyss = collect(keys(histogram))
	
	metrics = Dict()
	metrics["min_degrees"] = [minimum(keyss)]
	metrics["avg_degrees"] = [Graphs.ne(g) * 2 / Graphs.nv(g)]
    metrics["max_degrees"] = [maximum(keyss)]
	metrics["avg_edge_dist"] = [OpinionDiffusion.StatsBase.mean(OpinionDiffusion.get_edge_distances(g, voters))]
    metrics["clustering_coeff"] = [Graphs.global_clustering_coefficient(g)]
	#metrics["diameter"] = [Graphs.diameter(g)]
    
	#election results
	votes = get_votes(voters)
	metrics["avg_vote_length"] = [OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes])]
    metrics["unique_votes"] = [length(unique(votes))]
	metrics["election_matrix"] = [sum(abs.(get_counts(votes, can_count)), dims=1)]

    metrics["plurality_votings"] = [plurality_voting(votes, can_count, true)]
    metrics["borda_votings"] = [borda_voting(votes, can_count, true)]
    #metrics["copeland_votings"] = [copeland_voting(votes, can_count)]

	metrics["positions"] = [get_positions(voters, can_count)]
	
	return metrics
end

function update_metrics!(model, diffusion_metrics)
    g = get_social_network(model)
    voters = get_voters(model)
	candidates = get_candidates(model)
	can_count = length(candidates)
	
    dict = Graphs.degree_histogram(g)
    keyss = collect(keys(dict))
	
    push!(diffusion_metrics["min_degrees"], minimum(keyss))
    push!(diffusion_metrics["avg_degrees"], Graphs.ne(g) * 2 / Graphs.nv(g))
    push!(diffusion_metrics["max_degrees"], maximum(keyss))
	push!(diffusion_metrics["avg_edge_dist"], OpinionDiffusion.StatsBase.mean(OpinionDiffusion.get_edge_distances(g, voters)))
    push!(diffusion_metrics["clustering_coeff"], Graphs.global_clustering_coefficient(g))
    #push!(diffusion_metrics["diameter"], Graphs.diameter(g))
    
    votes = get_votes(voters)
	push!(diffusion_metrics["avg_vote_length"], OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes]))
    push!(diffusion_metrics["unique_votes"], length(unique(votes)))
	push!(diffusion_metrics["election_matrix"], sum(abs.(get_counts(votes, can_count)), dims=1))

    push!(diffusion_metrics["plurality_votings"], plurality_voting(votes, can_count, true))
    push!(diffusion_metrics["borda_votings"], borda_voting(votes, can_count, true))
    #push!(diffusion_metrics["copeland_votings"], copeland_voting(votes, can_count))
    push!(diffusion_metrics["positions"], get_positions(voters, can_count))
end

function metrics_vis(metrics, candidates, parties, exp_dir=Nothing)
    degrees = draw_range(metrics["min_degrees"], metrics["avg_degrees"], metrics["max_degrees"], title="Degree range", xlabel="Timestamp", ylabel="Degree", value_label="avg")

    plurality = draw_voting_res(candidates, parties, reduce(hcat, metrics["plurality_votings"])', "Plurality voting")
    borda = draw_voting_res(candidates, parties, reduce(hcat, metrics["borda_votings"])', "Borda voting")
    copeland = draw_voting_res(candidates, parties, reduce(hcat, metrics["copeland_votings"])', "Copeland voting")

    plots = Plots.plot(degrees, plurality, borda, copeland, layout = (2, 2), size = (669,900))
    
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

    Plots.plot!(plot, x, min, fillrange = max, fillalpha = 0.4, c = c, linewidth = 0, label=label)
    Plots.plot!(plot, x, value, linewidth = 3, label = label, c = c, linestyle=linestyle)
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
    Plots.plot!(plot, title=title, xlabel="Timestamp", ylabel="Value", yformatter = :plain)

    return plot
end

#___________________________________________________________________
# model
function model_vis(model, sampled_voter_ids, reduce_dim_config, clustering_config, interm_calcs=Dict())
    visualizations = []
    social_network = get_social_network(model)
    voters = get_voters(model)

    #voter visualization
    sampled_voters = voters[sampled_voter_ids]
    sampled_opinions = reduce(hcat, get_opinion(sampled_voters))

    projections = reduce_dims(sampled_opinions, reduce_dim_config)

    if haskey(interm_calcs, "prev_projections")
        unify_projections!(interm_calcs["prev_projections"], projections)
    end
    
    labels, clusters = clustering(sampled_voters, clustering_config, projections)
    if haskey(interm_calcs, "prev_clusters")
        unify_clusters!(interm_calcs["prev_clusters"], clusters)
        for (label, indices) in clusters
            labels[collect(indices)] .= label
        end
    end
    

    title = name(reduce_dim_config) * "_" * name(clustering_config) * "_" * string(length(sampled_voters))
    push!(visualizations, draw_voter_vis(projections, clusters, title))

    dens = KernelDensity.kde((projections[1, :], projections[2, :]))
    push!(visualizations, Plots.heatmap(dens, title="Voter map", c=Plots.cgrad([:blue, :white,:red, :yellow])))
    g = get_cluster_graph(model, clusters, labels, projections)
    cluster_metrics = nothing#cluster_graph_metrics(g, social_network, voters, 4)
    #println(modularity(g, labels))
    
    f = Figure()
    draw_voter_KDE!(f[1, 1], projections)
    draw_cluster_graph2(f[1, 1], g)
    push!(visualizations, f)

    push!(visualizations, draw_degree_distr(Graphs.degree_histogram(social_network)))

    push!(visualizations, draw_edge_distances(get_edge_distances(social_network, voters)))
    
    counts = get_counts(get_votes(sampled_voters), length(get_candidates(model)))
    if haskey(interm_calcs, "prev_counts")
        difference = counts - interm_calcs["prev_counts"]
        sum_difference = sum(abs.(difference), dims=1)
        push!(visualizations, heatmap(difference))
    end

    interm_calcs["prev_projections"] = projections
    interm_calcs["prev_clusters"] = clusters
    interm_calcs["prev_counts"] = counts
    return visualizations, interm_calcs
end

function draw_voter_vis(projections, clusters, title, exp_dir=Nothing, counter=[0])
    plot = Plots.plot()
    draw_voter_vis!(plot, projections, clusters, title, exp_dir, counter)
    
    return plot
end

function draw_voter_vis!(plot, projections, clusters, title, exp_dir=Nothing, counter=[0])
    cluster_colors = Colors.distinguishable_colors(maximum([cluster[1] for cluster in clusters]))

    for (label, indices) in clusters
        Plots.scatter!(plot, Tuple(eachrow(projections[:, collect(indices)])), c=cluster_colors[label], label=length(indices), alpha=0.4)
    end
   
    if exp_dir != Nothing
        Plots.savefig(plot, "$(exp_dir)/images/$(title)_$(counter[1]).png")
    end

    return plot
end

function draw_voter_KDE!(ax, projections)
    dens = KernelDensity.kde((projections[1, :], projections[2, :]))
    heatmap(ax, dens, colormap=[:blue, :white,:red, :yellow])
end

function draw_degree_distr(degree_distribution, exp_dir=Nothing, diff_counter=[0])
    plot = Plots.plot()
    draw_degree_distr!(plot, degree_distribution, exp_dir, diff_counter)

    return plot
end

function draw_degree_distr!(plot, degree_distribution, exp_dir=Nothing, diff_counter=[0])
    sorted = sort(degree_distribution)
    keyss = collect(keys(sorted))
    vals = collect(values(sorted))

    Plots.plot!(plot, keyss, vals, 
                        title = "Degree distribution",
                        legend = false,
                        xaxis=:log, yaxis=:log,
                        ylabel = "Num. of vertices (log)",
                        xlabel = "Degree (log)")

    if exp_dir != Nothing
        Plots.savefig(plot, "$(exp_dir)/images/degree_distribution_$(diff_counter[1]).png")
    end
end

function draw_voting_res(candidates, parties, result, title::String; linestyle=:solid, log_idx="")
    plot = Plots.plot()
    draw_voting_res!(plot, candidates, parties, result, title; linestyle=linestyle, log_idx="")

    return plot 
end

function draw_voting_res!(plot, candidates, parties, result, title::String; linestyle=:solid, log_idx="")
    names = [string(i) * " - " * parties[candidate.party] * " " * log_idx  for (i, candidate) in enumerate(candidates)]
    c = Colors.distinguishable_colors(size(result, 2))

    for (i, col) in enumerate(eachcol(result))
        draw_range!(plot, [x[2] for x in col], [x[3] for x in col], [x[4] for x in col], c=c[i], linestyle=linestyle, label=names[i])
        Plots.plot!(plot, title=title, xlabel="Timestamp", ylabel="Percentage", yformatter = :plain)
    end
end

function draw_heat_vis(projections, difference, title, exp_dir=Nothing, counter=[0])
    plot = Plots.plot()
    draw_heat_vis!(plot, projections, difference, title, exp_dir, counter)
    
    return plot
end

function draw_heat_vis!(plot, projections, difference, title, exp_dir=Nothing, counter=[0])
    Plots.scatter!(plot, Tuple(eachrow(projections)), marker_z=difference, title=title)
    
    if exp_dir != Nothing
        Plots.savefig(plot, "$(exp_dir)/images/$(title)_$(counter[1]).png")
    end

    return plot
end

function get_edge_distances(social_network, voters)
    distances = Vector{Float64}(undef, Graphs.ne(social_network))
    for (i, edge) in enumerate(Graphs.edges(social_network))
       distances[i] = get_distance(voters[Graphs.src(edge)], voters[Graphs.dst(edge)])
    end

    return distances
end

function get_edge_distances2(social_network, voters)
    distances = Vector{Any}(undef, Graphs.ne(social_network))
    for (i, edge) in enumerate(Graphs.edges(social_network))
       distances[i] = (Graphs.src(edge), Graphs.dst(edge), get_distance(voters[Graphs.src(edge)], voters[Graphs.dst(edge)]))
    end

    return distances
end

function draw_edge_distances(distances)
    plot = Plots.plot()
    draw_edge_distances!(plot, distances)

    return plot
end

function draw_edge_distances!(plot, distances)
    Plots.histogram!(plot, distances,
                         title = "Edge distance distribution",
                         nbins = 100,
                         xlims = (0.0, 1.0),
                         legend = false,
                         ylabel = "Num. of vertices",
                         xlabel = "Distance")
end

function election_summary(votes::Vector{Vote}, can_count::Int64)
	result = zeros(Float64, can_count, can_count)
	
	for vote in votes
        position = 1
        for bucket in vote
			for c in bucket
            	result[c, position] += 1.0 / length(bucket)
                position += 1
			end
        end
    end
	
	return result
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

"""
Loads all logs from one experiment and returns dictionary of visualizations
"""
function gather_vis(exp_dir, sampled_voter_ids, dim_reduction_config, clustering_config, )
    timestamps = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(exp_dir) if split(file, "_")[1] == "model"])
    visualizations = []
    interm_calc = Dict()

    for t in timestamps
        model_log = load_log(exp_dir, t)
        visualization, interm_calc = model_vis(model_log, sampled_voter_ids, dim_reduction_config, clustering_config, interm_calc)
        push!(visualizations, visualization)
        #push!(visualizations, stack_visualizations(model_vis2(model_log, sampled_voter_ids, dim_reduction_config, clustering_config)))
    end
    
    return visualizations
end

function stack_visualizations(visualizations)
    n = length(visualizations)
    return Plots.plot(visualizations... ,size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm, legend=true)
end