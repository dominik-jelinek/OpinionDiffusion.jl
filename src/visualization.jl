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

function visualize_metrics(election, timesteps::Vector{Metrics}, log_dir=Nothing)
    println("Analysis:")
    println()
    println("Candidates: ", size(database, 1))
    println("Voters: ", size(database, 2))
 
    averagePrefLengths = drawStat([step.averagePrefLength for step in steps], "Average pref. lengths")
    averageDistances = drawStat([step.averageDistance for step in steps], "Average distances")
    longestDistances = drawStat([step.longestDistance for step in steps], "Longest distances")
    edgeCounts = drawStat([step.edgeCount for step in steps], "Edge counts")
    
    #prefLengthDist = drawPrefLengthDistribution(database)
    #degreeDist = drawDegreeDistribution(g)
    plots = plot(averagePrefLengths, averageDistances, longestDistances, edgeCounts, layout = (2, 2), size = (980,1200))
    if logdir != Nothing
       savefig(plots, "$(logdir)/statistics.png")
    end
    display(plots)
end
 
function visualize_voting_rules(candidates, parties, steps::Vector{Metrics}, logdir=Nothing)
    #stackedResult = drawElectionResult(candidates, parties, getElectionResult(database))
 
    plurality = zeros(Float64, length(candidates), length(steps))
    for i in 1:length(steps)
       plurality[:, i] .= steps[i].pluralityVoting
    end
 
    borda = zeros(Float64, length(candidates), length(steps))
    for i in 1:length(steps)
       borda[:, i] .= steps[i].bordaVoting
    end
 
    plurality = drawVotingResult(candidates, parties, plurality', "Plurality count")
    borda = drawVotingResult(candidates, parties, borda', "Borda count")
    
    plots = plot(plurality, borda, layout = (1, 2), size = (1000,500))
    if logdir != Nothing
       savefig(plots, "$(logdir)/elections.png")
    end
    display(plots)
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
 
 function drawStat(values, title::String)
    plot(values,
    title=title,
    xlabel="Steps",
    ylabel="Value",
    xticks=(1:length(values)),
    linewidth = 3,
    legend = true,
    yformatter = :plain
    )
 end
 
 function drawVotingResult(candidates, parties, result, title::String)
    plot(result,
    title=title,
    xlabel="Steps",
    ylabel="Percentage",
    xticks=(1:size(result, 2)),
    #yticks=(1:size(result,1)),
    label = [candidate.name * "-" * parties[candidate.party] for candidate in candidates],
    linewidth = 3,
    legend = true,
    yformatter = :plain
    )
 end