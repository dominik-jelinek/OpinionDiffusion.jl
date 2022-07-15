abstract type Abstract_clustering_config <: Config end
@kwdef struct Kmeans_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end

function clustering(voters, clustering_config::Kmeans_clustering_config)
    opinions = reduce(hcat, get_opinion(voters))

    kmeans_res = Clustering.kmeans(opinions, clustering_config.cluster_count; maxiter=200)
    labels = kmeans_res.assignments
    
    clusters = clusterize(labels, clustering_config.cluster_count)
    
    return labels, clusters
end

@kwdef struct GM_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end

function clustering(voters, clustering_config::GM_clustering_config)
    opinions = reduce(hcat, get_opinion(voters))

    data_T = permutedims(opinions)
    gm = ScikitLearn.GaussianMixture(n_components=clustering_config.cluster_count).fit(data_T)
    labels = gm.predict(data_T) .+ 1
    
    clusters = clusterize(labels, clustering_config.cluster_count)
    
    return labels, clusters
end

@kwdef struct Party_clustering_config <: Abstract_clustering_config
    candidates::Vector{Candidate}
    parties_count::Int64
end

"""
Cluster voters based on highest ranked candidate
"""
function clustering(voters, clustering_config::Party_clustering_config)
    candidates = clustering_config.candidates

    labels = [candidates[iterate(get_vote(voter)[1])[1]].party for voter in voters] 
    
    clusters = clusterize(labels, clustering_config.parties_count)
    
    return labels, clusters
end

@kwdef struct DBSCAN_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end

function clustering(voters, clustering_config::DBSCAN_clustering_config)
    opinions = reduce(hcat, get_opinion(voters))

    res = Clustering.DBSCAN(opinions, clustering_config.cluster_count; maxiter=200)
    labels = res.assignments
    
    clusters = clusterize(labels, clustering_config.cluster_count)
    
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