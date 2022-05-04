function reduce_dim(sampled_opinions, reduce_dim_config)
    if reduce_dim_config.method == "PCA"
        config = reduce_dim_config.pca_config
        model = MultivariateStats.fit(MultivariateStats.PCA, sampled_opinions; maxoutdim=config.out_dim)
        projection = MultivariateStats.transform(model, sampled_opinions)
        println(MultivariateStats.mean(model))
        #println(MultivariateStats.projection(model))
        println(MultivariateStats.principalratio(model))
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

function ensemble_vis(experiment_names, sampled_voter_ids)
    #=
    Gather data from the logs of multiple diffusion experiments and visualize spreads
    =#
    for log in experiment_names
        model_log = load_log(logger.exp_dir, step)
        sampled_voters = model_log.voters[sampled_voter_ids]
        sampled_opinions = get_opinion(sampled_voters)

        projections = reduce_dim(sampled_opinions, reduce_dim_config)

        labels, clusters = clustering(sampled_opinions, candidates, length(parties), clustering_config)

        #heatmap
        difference = sampled_opinions - prev_sampled_opinions
        changes = vec(sum(abs.(difference), dims=1))
        println(sum(changes))
        draw_heat_vis(projections, changes, "Heat map")

        draw_voter_vis(projections, clusters, title)
        draw_edge_distances(get_edge_distances(model_log.social_network, model_log.voters))

        metrics_vis(metrics, candidates, parties)
    end
end

function init_metrics(model, can_count)
	metrics = Dict()
	histogram = Graphs.degree_histogram(model.social_network)
    keyss = collect(keys(histogram))
	
	metrics["min_degrees"] = [minimum(keyss)]
	metrics["avg_degrees"] = [Graphs.ne(model.social_network) * 2 / Graphs.Graphs.nv(model.social_network)]
    metrics["max_degrees"] = [maximum(keyss)]
 
    #election results
	votes = get_votes(model.voters)
    metrics["plurality_votings"] = [plurality_voting(votes, can_count, true)]
    metrics["borda_votings"] = [borda_voting(votes, can_count, true)]
    metrics["copeland_votings"] = [copeland_voting(votes, can_count)]
	
	return metrics
end

function update_metrics!(model, diffusion_metrics, can_count)
    g = social_network(model)
    voters = voters(model)

    dict = Graphs.degree_histogram(g)
    keyss = collect(keys(dict))
	
    push!(diffusion_metrics["min_degrees"], minimum(keyss))
    push!(diffusion_metrics["avg_degrees"], Graphs.ne(g) * 2 / Graphs.nv(g))
    push!(diffusion_metrics["max_degrees"], maximum(keyss))
    push!(diffusion_metrics["clustering_coeff"], Graphs.global_clustering_coefficient(g))
    
    votes = get_votes(voters)
	push!(diffusion_metrics["avg_vote_length"], StatsBase.mean([length(vote) for vote in votes]))
    push!(diffusion_metrics["unique_votes"], length(unique(votes)))
    
    mean_nei_dist = StatsBase.mean([StatsBase.mean(get_distance(voter, voters[Graphs.neighbors(g, voter.ID)])) for voter in voters])
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
        Plots.scatter!(plot, Tuple(eachrow(projections[:, idxes])), c=cluster_colors[i], label=length(clusters[i]))
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

function draw_range(min, value, max; title, xlabel, ylabel, value_label)
    degrees = Plots.plot(1:length(min), min, fillrange = max, fillalpha = 0.25, c = 1, linewidth = 3, 
    label = "range", legend = :topleft, title=title, xlabel=xlabel, ylabel=ylabel)
    
    Plots.plot!(degrees, 1:length(value), value, linewidth = 3, label = value_label)
    
    return degrees
end

function draw_metric(values, title::String)
    Plots.plot(values,
    title=title,
    xlabel="Diffusions",
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
    legend = false,
    yformatter = :plain
    )
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

