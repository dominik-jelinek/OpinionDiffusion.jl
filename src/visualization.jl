#=
function draw_voter_visualization(sampled_opinions, sampled_election, candidates, parties, expConfig, expDir, counter) 
    projections = reduce_dim(sampled_opinions, expConfig["reduceDimConfig"])
    _, clusters = clustering(sampled_opinions, sampled_election, candidates, parties, expConfig["clusteringConfig"])
    
    draw_voter_visualization(projections, clusters, expConfig["reduceDimConfig"]["method"], expConfig["clusteringConfig"]["method"], expDir, counter)
end
=#

function draw_voter_visualization(projections, clusters, dim_red_method, clust_method, exp_dir=Nothing, counter=[0])
    cluster_colors  = distinguishable_colors(length(clusters), colorant"blue")
    title = dim_red_method * "_" * clust_method * "_" * string(size(projections, 2))

    idxes = collect(clusters[1])
    plot = scatter(Tuple(eachrow(projections[:, idxes])), c=cluster_colors[1], label=length(clusters[1]), title=title)
    for i in 2:length(clusters)
        idxes = collect(clusters[i])
        scatter!(plot, Tuple(eachrow(projections[:, idxes])), c=cluster_colors[i], label=length(clusters[i]))
    end
   
    if exp_dir != Nothing
        savefig(plot, "$(exp_dir)/images/$(title)_$(counter[1]).png")
    end

    return plot
end

function draw_voter_visualization(voters, sampled_voter_ids, candidates::Vector{Candidate}, parties::Vector{String}, exp_dir::String, diff_counter, voter_visualization_config)
    sampled_voters = voters[sampled_voter_ids]
    sampled_opinions = get_opinions(sampled_voters)

    projections = reduce_dim(sampled_opinions, voter_visualization_config["reduce_dim_config"])
    _, clusters = clustering(sampled_opinions, candidates, voter_visualization_config["clustering_config"])

    draw_voter_visualization(projections, clusters, voter_visualization_config["reduce_dim_config"]["method"], voter_visualization_config["clustering_config"]["method"], exp_dir, diff_counter)
end

function reduce_dim(sampled_opinions, reduce_dim_Config)
    if reduce_dim_Config["method"] == "PCA"
        config = reduce_dim_Config["PCA"]
        model = MultivariateStats.fit(PCA, sampled_opinions; maxoutdim=config["out_dim"])
        projection = MultivariateStats.transform(model, sampled_opinions)

    elseif reduceDimConfig["method"] == "tsne"
        config = reduce_dim_Config["tsne"]
        projection = permutedims(tsne(sampled_opinions, config["out_dim"], config["reduce_dims"], config["max_iter"], config["perplexity"]))
    else
        error("Unknown dimensionality reduction method")
    end

    return projection
end

function visualize_metrics(metrics, candidates, parties, exp_dir=Nothing)
    degrees = draw_range(metrics.min_degrees, metrics.avg_degrees, metrics.max_degrees, title="Degree range", xlabel="Diffusions", ylabel="Degree", value_label="avg")

    plurality = drawVotingResult(candidates, parties, reduce(hcat, metrics.plurality_votings)', "Plurality voting")
    borda = drawVotingResult(candidates, parties, reduce(hcat, metrics.borda_votings)', "Borda voting")
    copeland = drawVotingResult(candidates, parties, reduce(hcat, metrics.copeland_votings)', "Copeland voting")

    plots = plot(degrees, plurality, borda, copeland, layout = (2, 2), size = (980,1200))
    
    if exp_dir != Nothing
        savefig(plots, "$(exp_dir)/images/metrics.png")
    end

    return plots
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

function draw_degree_distribution(dict, exp_dir=Nothing, diff_counter=[0])
    keyss = collect(keys(dict))
    vals = collect(values(dict))

    plot = histogram(keyss,
                        weights = vals,
                        bins = length(keyss),
                        title = "Degree distribution",
                        legend = false,
                        ylabel = "Num. of vertices",
                        xlabel = "Num. of edges")

    if exp_dir != Nothing
        savefig(plot, "$(exp_dir)/images/degree_distribution_$(diff_counter[1]).png")
    end

    return plot
 end
 

 
 function drawVotingResult(candidates, parties, result, title::String)
    names = [candidate.name * " - " * parties[candidate.party] for candidate in candidates]
    plot(result,
    title=title,
    xlabel="Diffusions",
    ylabel="Percentage",
    label = reshape(names, 1, length(names)),
    linewidth = 3,
    legend = false,
    yformatter = :plain
    )
 end