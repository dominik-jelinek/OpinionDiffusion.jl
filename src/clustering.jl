#=For kendall encoded
function clustering(sampled_opinions, sampled_election, candidates, parties, clusteringConfig)
    
    if clusteringConfig["method"] == "Party"
        labels = if sampled_election[1, :]
        clusters = clusterize(labels, candidates, parties)
        
    elseif clusteringConfig["method"] == "K-means"
        KmeansRes = kmeans(sampled_opinions, clusteringConfig["K-means"]["clusterCount"]; maxiter=200)
        labels = KmeansRes.assignments
        clusters = clusterize(labels)
        
    elseif clusteringConfig["method"] == "GM"
        data_T = permutedims(sampled_opinions)
        gm = GaussianMixture(n_components=clusteringConfig["GM"]["clusterCount"]).fit(data_T)
        labels = gm.predict(data_T) .+ 1
        clusters = clusterize(labels)
    else
        error("Unknown clustering method")
    end
    
    return labels, clusters
end
=#
function clustering(sampled_opinions, candidates, clustering_config)
    
    if clustering_config["method"] == "Party"
        labels = [candidates[argmin(col)].party for col in eachcol(sampled_opinions)] 
    elseif clustering_config["method"] == "K-means"
        kmeans_res = Clustering.kmeans(sampled_opinions, clustering_config["K-means"]["cluster_count"]; maxiter=200)
        labels = kmeans_res.assignments 
    elseif clustering_config["method"] == "GM"
        data_T = permutedims(sampled_opinions)
        gm = ScikitLearn.GaussianMixture(n_components=clustering_config["GM"]["cluster_count"]).fit(data_T)
        labels = gm.predict(data_T) .+ 1
    else
        error("Unknown clustering method")
    end

    clusters = clusterize(labels)
    return labels, clusters
end

function clusterize(labels)
    cluster_count = length(unique(labels))
    clusters = Vector{Set{Int64}}(undef, cluster_count)
    for i in 1:cluster_count
        clusters[i] = Set(findall(x->x==i, labels))
    end

    return clusters
end

function unify_labels!(template_clusters, clusters)
    for i in 1:length(template_clusters)
        best = 0.0
        idx = i
        for j in i:length(clusters)
            similarity = length(intersect(template_clusters[i], clusters[j])) 
            if similarity > best
                best = similarity
                idx = j
            end
        end
        clusters[i], clusters[idx] = clusters[idx], clusters[i]
    end
end