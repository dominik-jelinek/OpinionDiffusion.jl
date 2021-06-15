#For kendall encoded
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

function clustering(sampled_opinions, candidates, clusteringConfig)
    
    if clusteringConfig["method"] == "Party"
        labels = [candidates[argmin(col)].party for col in eachcol(sampled_opinions)]
        clusters = clusterize(labels)
        
    elseif clusteringConfig["method"] == "K-means"
        KmeansRes = kmeans(sampled_opinions, clusteringConfig["K-means"]["cluster_count"]; maxiter=200)
        labels = KmeansRes.assignments
        clusters = clusterize(labels)
        
    elseif clusteringConfig["method"] == "GM"
        data_T = permutedims(sampled_opinions)
        gm = GaussianMixture(n_components=clusteringConfig["GM"]["cluster_count"]).fit(data_T)
        labels = gm.predict(data_T) .+ 1
        clusters = clusterize(labels)
    else
        error("Unknown clustering method")
    end
    
    return labels, clusters
end

function clusterize(labels)
    cluster_count = length(unique(labels))
    clusters = Vector{Set{Int}}(undef, cluster_count)
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