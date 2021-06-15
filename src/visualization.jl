function visualize_voters(sampled_opinions, sampled_election, candidates, parties, expConfig, expDir, counter) 
    projections = reduce_dim(sampled_opinions, expConfig["reduceDimConfig"])
    _, clusters = clustering(sampled_opinions, sampled_election, candidates, parties, expConfig["clusteringConfig"])
    
    visualize_voters(projections, clusters, expConfig["reduceDimConfig"]["method"], expConfig["clusteringConfig"]["method"], expDir, counter)
end

function visualize_voters(sampled_opinions, candidates, parties, expConfig, expDir, counter) 
    projections = reduceDim(sampled_opinions, expConfig["reduceDimConfig"])
    _, clusters = clustering(sampled_opinions, candidates, parties, expConfig["clusteringConfig"])
    
    visualize_voters(projections, clusters, expConfig["reduceDimConfig"]["method"], expConfig["clusteringConfig"]["method"], expDir, counter)
end

function visualize_voters(projections, clusters, dim_red_method, clust_method, exp_dir=Nothing, counter=0)
    cluster_colors  = distinguishable_colors(length(clusters), colorant"blue")
    title = dim_red_method * "_" * clust_method * "_" * string(size(projections, 2))

    idxes = collect(clusters[1])
    plot = scatter(Tuple(eachrow(projections[:, idxes])), c=cluster_colors[1], label=length(clusters[1]), title=title)
    for i in 2:length(clusters)
        idxes = collect(clusters[i])
        scatter!(plot, Tuple(eachrow(projections[:, idxes])), c=cluster_colors[i], label=length(clusters[i]))
    end
   
    if logdir != Nothing
        savefig(plot, "$(exp_dir)/voters/$(title)_$(counter).png")
    end
    display(plot)
end

function reduce_dim(sampled_opinions, reduce_dim_Config)
    if reduce_dim_Config["method"] == "PCA"
        config = reduce_dim_Config["PCA"]
        model = MultivariateStats.fit(PCA, sampled_opinions; maxoutdim=config["outDim"])
        projection = MultivariateStats.transform(model, sampled_opinions)

    elseif reduceDimConfig["method"] == "tsne"
        config = reduce_dim_Config["tsne"]
        projection = permutedims(tsne(sampled_opinions, config["outDim"], config["reduce_dims"], config["max_iter"], config["perplexity"]))
    else
        error("Unknown dimensionality reduction method")
    end

    return projection
end

function visualize_metrics(experiment)
    metrics = experiment.diffusion_metrics
    degrees = draw_range(metrics.min_degrees, metrics.avg_degrees, metrics.max_degrees, title="Degree range", xlabel="Diffusions", ylabel="Degree", value_label="avg")

    plurality = drawVotingResult(experiment.candidates, experiment.parties, reduce(hcat, metrics.plurality_votings)', "Plurality voting")
    borda = drawVotingResult(experiment.candidates, experiment.parties, reduce(hcat, metrics.borda_votings)', "Borda voting")
    copeland = drawVotingResult(experiment.candidates, experiment.parties, reduce(hcat, metrics.copeland_votings)', "Copeland voting")

    plots = plot(degrees, plurality, borda, copeland, layout = (2, 2), size = (980,1200))
    
    savefig(plots, "$(experiment.exp_dir)/images/metrics.png")
    display(plots)
end

function draw_range(min, value, max; title, xlabel, ylabel, value_label)
    degrees = plot(1:length(min), min, fillrange = max, fillalpha = 0.25, c = 1, linewidth = 3, 
    label = "range", legend = :topleft, title=title, xlabel=xlabel, ylabel=ylabel)
    
    plot!(degrees, 1:length(value), value, linewidth = 3, label = value_label)
    
    return degrees
end

function draw_metric(values, title::String)
    plot(values,
    title=title,
    xlabel="Diffusions",
    ylabel="Value",
    xticks=(1:length(values)),
    linewidth = 3,
    legend = true,
    yformatter = :plain
    )
end

function drawDegreeDistribution(g)
    dict = degree_histogram(g)
    keyss = collect(keys(dict))
    vals = collect(values(dict))
 
    histogram(keyss,
             weights = vals,
             bins = length(keyss),
             title = "Degree distribution",
             legend = false,
             ylabel = "Num. of vertices",
             xlabel = "Num. of edges")
 end
 

 
 function drawVotingResult(candidates, parties, result, title::String)
    names = [candidate.name * " - " * parties[candidate.party] for candidate in candidates]
    plot(result,
    title=title,
    xlabel="Diffusions",
    ylabel="Percentage",
    xticks=(1:size(result, 1)),
    label = reshape(names, 1, length(names)),
    linewidth = 3,
    legend = false,
    yformatter = :plain
    )
 end