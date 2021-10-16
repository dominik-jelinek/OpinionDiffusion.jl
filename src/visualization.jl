function get_voter_vis(voters, sampled_voter_ids, candidates::Vector{Candidate}, voter_visualization_config)
    sampled_voters = voters[sampled_voter_ids]
    sampled_opinions = get_opinions(sampled_voters)

    projections = reduce_dim(sampled_opinions, voter_visualization_config.reduce_dim_config)
    labels, clusters = clustering(sampled_opinions, candidates, voter_visualization_config.clustering_config)
    return projections, labels, clusters
end

function reduce_dim(sampled_opinions, reduce_dim_Config)
    if reduce_dim_Config.method == "PCA"
        config = reduce_dim_Config.pca_config
        model = MultivariateStats.fit(MultivariateStats.PCA, sampled_opinions; maxoutdim=config.out_dim)
        projection = MultivariateStats.transform(model, sampled_opinions)

    elseif reduceDimConfig.method == "tsne"
        config = reduce_dim_Config.tsne_config
        projection = permutedims(TSne.tsne(sampled_opinions, config.out_dim, config.reduce_dims, config.max_iter, config.perplexity))
    else
        error("Unknown dimensionality reduction method")
    end

    return projection
end

function draw_voter_vis(projections, clusters, voter_visualization_config, exp_dir=Nothing, counter=[0])
    cluster_colors  = Colors.distinguishable_colors(length(clusters))
    title = voter_visualization_config.reduce_dim_config.method * "_" * voter_visualization_config.clustering_config.method * "_" * string(size(projections, 2))

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

function get_edge_distances(social_network, voters)
    distances = Vector{Float64}(undef, LightGraphs.ne(social_network))
    for (i, edge) in enumerate(LightGraphs.edges(social_network))
       distances[i] = get_distance(voters[LightGraphs.src(edge)], voters[LightGraphs.dst(edge)])
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

function metrics_vis(metrics, candidates, parties, exp_dir=Nothing)
    degrees = draw_range(metrics.min_degrees, metrics.avg_degrees, metrics.max_degrees, title="Degree range", xlabel="Diffusions", ylabel="Degree", value_label="avg")

    plurality = draw_voting_res(candidates, parties, reduce(hcat, metrics.plurality_votings)', "Plurality voting")
    borda = draw_voting_res(candidates, parties, reduce(hcat, metrics.borda_votings)', "Borda voting")
    copeland = draw_voting_res(candidates, parties, reduce(hcat, metrics.copeland_votings)', "Copeland voting")

    plots = Plots.plot(degrees, plurality, borda, copeland, layout = (2, 2), size = (980,1200))
    
    if exp_dir != Nothing
        Plots.savefig(plots, "$(exp_dir)/images/metrics.png")
    end

    return plots
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
                        nbins = length(keyss),
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