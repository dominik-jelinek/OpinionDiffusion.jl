function clustering(voters, candidates, parties_count, clustering_config)

    if clustering_config.method == "Party"
        return party_clustering(voters, candidates, parties_count)
    elseif clustering_config.method == "K-means"
        opinions = reduce(hcat, get_opinion(voters))
        kmeans_res = Clustering.kmeans(opinions, clustering_config.kmeans_config.cluster_count; maxiter=200)
        labels = kmeans_res.assignments
        clusters = clusterize(labels, clustering_config.kmeans_config.cluster_count)
    elseif clustering_config.method == "GM"
        data_T = permutedims(get_opinion(voters))
        gm = ScikitLearn.GaussianMixture(n_components=clustering_config.gm_config.cluster_count).fit(data_T)
        labels = gm.predict(data_T) .+ 1
        clusters = clusterize(labels, clustering_config.gm_config.cluster_count)
    elseif clustering_config.method == "DBSCAN"
        res = Clustering.DBSCAN(get_opinion(voters), clustering_config.kmeans_config.cluster_count; maxiter=200)
    else
        error("Unknown clustering method")
    end

    
    return labels, clusters
end

"""
Cluster voters based on highest ranked candidate
"""
function party_clustering(voters, candidates, parties_count)
    labels = [candidates[get_vote(voter)[1][1]].party for voter in voters] 
    clusters = clusterize(labels, parties_count)
    
    return labels, clusters
end

"""
Split labels into clusters
"""
function clusterize(labels, cluster_count)
    clusters = Vector{Set{Int64}}(undef, cluster_count)
    for i in 1:cluster_count
        clusters[i] = Set(findall(x->x==i, labels))
    end

    return clusters
end

"""
Finds the best bijection from clusters to template_clusters based on set overlap
"""
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

function unify_projections!(projections, x_projections, y_projections, x_treshold=1.0, y_treshold=1.0)
    if sum(projections[1, 1:length(x_projections)] - x_projections) < x_treshold
		@views row = projections[1, :] 
		row .*= -1.0
	end
	if sum(projections[2, 1:length(y_projections)] - y_projections) < y_treshold
		@views row = projections[2, :] 
		row .*= -1.0
	end
end