function reduce_dim(sampled_opinions, reduce_dim_config)
    if reduce_dim_config.method == "PCA"
        config = reduce_dim_config.pca_config
        model = MultivariateStats.fit(MultivariateStats.PCA, sampled_opinions; maxoutdim=config.out_dim)
        projection = MultivariateStats.transform(model, sampled_opinions)
        #println(MultivariateStats.loadings(model))
        #println(MultivariateStats.projection(model))
        #println(MultivariateStats.principalratio(model))
        #println(MultivariateStats.principalvars(model))
        #println(MultivariateStats.tresidualvar(model))
    elseif reduce_dim_config.method == "Tsne"
        config = reduce_dim_config.tsne_config
        projection = permutedims(TSne.tsne(permutedims(sampled_opinions), config.out_dim, config.reduce_dims, config.max_iter, config.perplexity))
    elseif reduce_dim_config.method == "MDS"
        model = MultivariateStats.fit(MDS, sampled_opinions; maxoutdim=2, distances=false)
        println(model)
        println(methods(predict))
        projection = predict(model, sampled_opinions)
	
        #projection = MultivariateStats.transform(model, sampled_opinions)
    else
        error("Unknown dimensionality reduction method")
    end

    return projection
end

"""
Gather data from the logs of multiple diffusion experiments and visualize spreads
"""
function gather_metrics(ens_metrics)
    if length(ens_metrics) == 0
        return
    end

    res = Dict()
    for metric in keys(ens_metrics[1])
        #println(metric)
        matrix = transpose(hcat([run[metric] for run in ens_metrics]...))
        if matrix[1,1] isa Number
            res[metric] = [Statistics.quantile(col, [0.0, 0.25, 0.5, 0.75, 1.0]) for col in eachcol(matrix)]
        else
            res[metric] = []
            for col in eachcol(matrix)                
                matrix_vect = vcat(col...)
                push!(res[metric], [Statistics.quantile(col, [0.0, 0.25, 0.5, 0.75, 1.0]) for col in eachcol(matrix_vect)])
            end

        end
        #display(res[metric])

    end

    return res
end

"""
Loads all logs from one experiment and returns dictionary of visualizations
"""
function gather_vis(exp_dir, sampled_voter_ids, reduce_dim_config, clustering_config, parties, candidates)
    last = last_log_idx(exp_dir)
    visualizations = Dict("heatmaps"=>[], "voters"=>[], "distances"=>[])
    title = reduce_dim_config.method * "_" * clustering_config.method * "_" * string(length(sampled_voter_ids))
    prev_sampled_opinions = nothing
    x_projections, y_projections, max_x, min_x, max_y, min_y = nothing,nothing,nothing,nothing,nothing,nothing
    for step in 0:last
        model_log = load_log(exp_dir, step)
        sampled_voters = get_voters(model_log)[sampled_voter_ids]
        sampled_opinions = reduce(hcat, get_opinion(sampled_voters))

        projections = reduce_dim(sampled_opinions, reduce_dim_config)
        
        labels, clusters = clustering(sampled_voters, candidates, length(parties), clustering_config)

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
    degrees = draw_range(metrics["min_degrees"], metrics["avg_degrees"], metrics["max_degrees"], title="Degree range", xlabel="Diffusions", ylabel="Degree", value_label="avg")

    plurality = draw_voting_res(candidates, parties, reduce(hcat, metrics["plurality_votings"])', "Plurality voting")
    borda = draw_voting_res(candidates, parties, reduce(hcat, metrics["borda_votings"])', "Borda voting")
    copeland = draw_voting_res(candidates, parties, reduce(hcat, metrics["copeland_votings"])', "Copeland voting")

    plots = Plots.plot(degrees, plurality, borda, copeland, layout = (2, 2), size = (669,900))
    
    if exp_dir != Nothing
        Plots.savefig(plots, "$(exp_dir)/images/metrics.png")
    end

    return plots
end

function draw_voter_vis(projections, clusters, title, exp_dir=Nothing, counter=[0])
    cluster_colors  = Colors.distinguishable_colors(length(clusters))

    idxes = collect(clusters[1])
    plot = Plots.scatter(Tuple(eachrow(projections[:, idxes])), c=cluster_colors[1], label=length(clusters[1]), title=title)
    for i in 2:length(clusters)
        idxes = collect(clusters[i])
        Plots.scatter!(plot, Tuple(eachrow(projections[:, idxes])), c=cluster_colors[i], label=length(clusters[i]), alpha=0.6)
    end
   
    if exp_dir != Nothing
        Plots.savefig(plot, "$(exp_dir)/images/$(title)_$(counter[1]).png")
    end

    return plot
end

function draw_heat_vis(projections, difference, title, exp_dir=Nothing, counter=[0])
    plot = Plots.scatter(Tuple(eachrow(projections)), marker_z=difference, title=title)
    
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

function draw_edge_distances(distances)
    Plots.histogram(distances,
                         title = "Edge distance distribution",
                         legend = false,
                         ylabel = "Num. of vertices",
                         xlabel = "Distance")
end

function draw_range!(plot, min, value, max; c=1, label)
    Plots.plot!(plot, 1:length(min), min, fillrange = max, fillalpha = 0.25, c = c, linewidth = 0, label=label)
    Plots.plot!(plot, 1:length(value), value, linewidth = 3, label = label, c = c)
    
    return plot
end

function draw_metric!(plot, values, title::String)
    draw_range!(plot, [x[2] for x in values], [x[3] for x in values], [x[4] for x in values], label=title)
    Plots.plot!(plot, title=title, xlabel="t", ylabel="Value", yformatter = :plain)

    return plot
end

function draw_metric(values, title::String)
    Plots.plot(values,
    title=title,
    xlabel="t",
    ylabel="Value",
    xticks=(1:length(values)),
    linewidth = 3,
    legend = true,
    yformatter = :plain
    )
end

function draw_degree_distr(dict, exp_dir=Nothing, diff_counter=[0])
    keyss = collect(keys(dict))
    vals = collect(values(dict))

    plot = Plots.histogram(keyss,
                        weights = vals,
                        nbins = maximum(keyss),
                        title = "Degree distribution",
                        legend = false,
                        ylabel = "Num. of vertices",
                        xlabel = "Num. of edges")

    if exp_dir != Nothing
        Plots.savefig(plot, "$(exp_dir)/images/degree_distribution_$(diff_counter[1]).png")
    end

    return plot
end
 
function draw_voting_res(candidates, parties, result, title::String)
    names = [candidate.name * " - " * parties[candidate.party] for candidate in candidates]
    #party_colors = Colors.distinguishable_colors(length())[parties[candidate.party] for candidate in candidates]
    Plots.plot(result,
    title=title,
    xlabel="Diffusions",
    ylabel="Percentage",
    label = reshape(names, 1, length(names)),
    linewidth = 3,
    yformatter = :plain
    )
end

function draw_voting_res!(plot, candidates, parties, result, title::String)
    names = [string(i) * " - " * parties[candidate.party] for (i, candidate) in enumerate(candidates)]
    c = Colors.distinguishable_colors(size(result, 2))

    for (i, col) in enumerate(eachcol(result))
        draw_range!(plot, [x[2] for x in col], [x[3] for x in col], [x[4] for x in col], c=c[i], label=names[i])
        Plots.plot!(plot, title=title, xlabel="t", ylabel="Percentage", yformatter = :plain)
    end
    
end

function ego(social_network, node_id, depth)
    neighs = Graphs.neighbors(social_network, node_id)
    ego_nodes = Set(neighs)
    front = Set(neighs)
    for i in 1:depth - 1
        new_front = Set()
        for voter in front
            union!(new_front, Graphs.neighbors(social_network, voter))
        end
        front = setdiff(new_front, ego_nodes)
        union!(ego_nodes, front)
    end

    return induced_subgraph(social_network, ego_nodes)
end

