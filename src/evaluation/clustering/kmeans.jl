@kwdef struct Kmeans_clustering_config <: Abstract_clustering_config
	cluster_count::Int64
	automatic_k::Bool = false
end
name(config::Kmeans_clustering_config) = "K-means"

function clustering(voters, clustering_config::Kmeans_clustering_config, projections=nothing)
	opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

	best_k = clustering_config.automatic_k ? best_k_silhouettes(opinions, clustering_config.cluster_count) : clustering_config.cluster_count
	kmeans_res = Clustering.kmeans(opinions, best_k)
	labels = kmeans_res.assignments

	clusters = clusterize(labels)

	return labels, clusters
end