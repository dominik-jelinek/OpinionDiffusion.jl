function reduce_dim(sampled_opinions, reduce_dim_Config)
    if reduce_dim_Config.method == "PCA"
        config = reduce_dim_Config.pca_config
        model = MultivariateStats.fit(MultivariateStats.PCA, sampled_opinions; maxoutdim=config.out_dim)
        projection = MultivariateStats.transform(model, sampled_opinions)
        println(MultivariateStats.mean(model))
        #println(MultivariateStats.projection(model))
        println(MultivariateStats.principalratio(model))
        #println(MultivariateStats.principalvars(model))
        #println(MultivariateStats.tresidualvar(model))
    elseif reduceDimConfig.method == "tsne"
        config = reduce_dim_Config.tsne_config
        projection = permutedims(TSne.tsne(sampled_opinions, config.out_dim, config.reduce_dims, config.max_iter, config.perplexity))
    else
        error("Unknown dimensionality reduction method")
    end

    return projection
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
