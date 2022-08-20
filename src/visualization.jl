# METRICS

function init_metrics(model, can_count)
    g = get_social_network(model)
    voters = get_voters(model)

	histogram = Graphs.degree_histogram(g)
    keyss = collect(keys(histogram))
	
	metrics = Dict()
	metrics["min_degrees"] = [minimum(keyss)]
	metrics["avg_degrees"] = [Graphs.ne(g) * 2 / Graphs.nv(g)]
    metrics["max_degrees"] = [maximum(keyss)]
    metrics["clustering_coeff"] = [Graphs.global_clustering_coefficient(g)]

    #election results
	votes = get_votes(voters)
	metrics["avg_vote_length"] = [StatsBase.mean([length(vote) for vote in votes])]
    metrics["mean_nei_dist"] = [StatsBase.mean([StatsBase.mean(get_distance(voter, voters[Graphs.neighbors(g, voter.ID)])) for voter in voters])]
    metrics["unique_votes"] = [length(unique(votes))]

    metrics["plurality_votings"] = [plurality_voting(votes, can_count, true)]
    metrics["borda_votings"] = [borda_voting(votes, can_count, true)]
    metrics["copeland_votings"] = [copeland_voting(votes, can_count)]
	
	return metrics
end

function update_metrics!(model, diffusion_metrics, can_count)
    g = get_social_network(model)
    voters = get_voters(model)

    dict = Graphs.degree_histogram(g)
    keyss = collect(keys(dict))
	
    push!(diffusion_metrics["min_degrees"], minimum(keyss))
    push!(diffusion_metrics["avg_degrees"], Graphs.ne(g) * 2 / Graphs.nv(g))
    push!(diffusion_metrics["max_degrees"], maximum(keyss))
    push!(diffusion_metrics["clustering_coeff"], Graphs.global_clustering_coefficient(g))
    
    votes = get_votes(voters)
	push!(diffusion_metrics["avg_vote_length"], StatsBase.mean([length(vote) for vote in votes]))    
    push!(diffusion_metrics["mean_nei_dist"], StatsBase.mean([StatsBase.mean(get_distance(voter, voters[Graphs.neighbors(g, voter.ID)])) for voter in voters]))
    push!(diffusion_metrics["unique_votes"], length(unique(votes)))

    push!(diffusion_metrics["plurality_votings"], plurality_voting(votes, can_count, true))
    push!(diffusion_metrics["borda_votings"], borda_voting(votes, can_count, true))
    push!(diffusion_metrics["copeland_votings"], copeland_voting(votes, can_count))
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
# MODEL

function model_vis(model, sampled_voter_ids, reduce_dim_config, clustering_config)
    social_network = get_social_network(model)
    voters = get_voters(model)

    #voter visualization
    sampled_voters = voters[sampled_voter_ids]
    sampled_opinions = reduce(hcat, get_opinion(sampled_voters))

    projections = reduce_dims(sampled_opinions, reduce_dim_config)
        
    labels, clusters = clustering(sampled_voters, clustering_config)

    n = 3
    plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)

    title = name(reduce_dim_config) * "_" * name(clustering_config) * "_" * string(length(sampled_voter_ids))
    draw_voter_vis!(plot[1, 1], projections, clusters, title)

    draw_degree_distr!(plot[2, 1], Graphs.degree_histogram(social_network))

    draw_edge_distances!(plot[3, 1], get_edge_distances(social_network, voters))

    return plot
end

function model_vis2(model, sampled_voter_ids, reduce_dim_config, clustering_config)
    visualizations = []
    social_network = get_social_network(model)
    voters = get_voters(model)

    #voter visualization
    #sampled_voters = voters[sampled_voter_ids]
    sampled_opinions = reduce(hcat, get_opinion(voters))

    projections = reduce_dims(sampled_opinions, reduce_dim_config)
        
    labels, clusters = clustering(voters, clustering_config)
    title = name(reduce_dim_config) * "_" * name(clustering_config) * "_" * string(length(voters))
    push!(visualizations, draw_voter_vis(projections, clusters, title))

    g = get_cluster_graph(model, clusters, labels, projections)
    println(modularity(g, labels))
    
    push!(visualizations, draw_cluster_graph(g))

    push!(visualizations, draw_degree_distr(Graphs.degree_histogram(social_network)))

    push!(visualizations, draw_edge_distances(get_edge_distances(social_network, voters)))

    return visualizations
end

function draw_voter_vis(projections, clusters, title, exp_dir=Nothing, counter=[0])
    plot = Plots.plot()
    draw_voter_vis!(plot, projections, clusters, title, exp_dir, counter)
    
    return plot
end

function draw_voter_vis!(plot, projections, clusters, title, exp_dir=Nothing, counter=[0])
    cluster_colors  = Colors.distinguishable_colors(length(clusters))

    idxes = collect(clusters[1])
    Plots.scatter!(plot, Tuple(eachrow(projections[:, idxes])), c=cluster_colors[1], label=length(clusters[1]), title=title)
    for i in 2:length(clusters)
        idxes = collect(clusters[i])
        Plots.scatter!(plot, Tuple(eachrow(projections[:, idxes])), c=cluster_colors[i], label=length(clusters[i]), alpha=0.4)
    end
   
    if exp_dir != Nothing
        Plots.savefig(plot, "$(exp_dir)/images/$(title)_$(counter[1]).png")
    end

    return plot
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
                         nbins = 20,
                         legend = false,
                         ylabel = "Num. of vertices",
                         xlabel = "Distance")
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
        matrix = transpose(hcat([run[metric] for run in ens_metrics]...))
        
        if matrix[1, 1] isa Number #number
            res[metric] = [Statistics.quantile(col, [0.0, 0.25, 0.5, 0.75, 1.0]) for col in eachcol(matrix)]
        
        else
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
function gather_vis(exp_dir, sampled_voter_ids, dim_reduction_config, clustering_config, parties, candidates)
    last = last_log_idx(exp_dir)
    visualizations = Dict("heatmaps"=>[], "voters"=>[], "distances"=>[])
    title = reduce_dim_config.method * "_" * clustering_config.method * "_" * string(length(sampled_voter_ids))
    prev_sampled_opinions = nothing
    x_projections, y_projections, max_x, min_x, max_y, min_y = nothing,nothing,nothing,nothing,nothing,nothing
    for step in 0:last
        model_log = load_log(exp_dir, step)
        sampled_voters = get_voters(model_log)[sampled_voter_ids]
        sampled_opinions = reduce(hcat, get_opinion(sampled_voters))

        projections = reduce_dims(sampled_opinions, dim_reduction_config)
        
        labels, clusters = clustering(sampled_voters, clustering_config)

        #heatmap
        if step > 0
            unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)

            difference = sampled_opinions - prev_sampled_opinions
            changes = vec(sum(abs.(difference), dims=1))
            println(sum(changes))
            push!(visualizations["heatmaps"], draw_heat_vis(projections, changes, "Heat map"))
        end
        prev_projections = projections
        x_projections = prev_projections[1, 1:length(candidates)]
    	y_projections = prev_projections[2, 1:length(candidates)]
        min_x, max_x = minimum(x_projections), maximum(x_projections) 
	    min_y, max_y = minimum(y_projections), maximum(y_projections)
        
        prev_sampled_opinions = sampled_opinions

        push!(visualizations["voters"], draw_voter_vis(projections, clusters, title))
        push!(visualizations["distances"], draw_edge_distances(get_edge_distances(model_log.social_network, model_log.voters)))
    end

    return visualizations
end


"""
Loads all logs from one experiment and returns dictionary of visualizations
"""
function gather_vis2(exp_dir, sampled_voter_ids, dim_reduction_config, clustering_config)
    timestamps = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(exp_dir) if split(file, "_")[1] == "model"])
    visualizations = []

    for t in timestamps
        model_log = load_log(exp_dir, t) 
        push!(visualizations, stack_visualizations(model_vis2(model_log, sampled_voter_ids, dim_reduction_config, clustering_config)))
        
    end
    
    return visualizations
end

function stack_visualizations(visualizations)
    n = length(visualizations)
    
    return Plots.plot(visualizations... ,size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm, legend=true)
end