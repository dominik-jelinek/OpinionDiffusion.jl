@kwdef struct Kmeans_clustering_config <: Abstract_clustering_config
	cluster_count::Int64
	automatic_k::Bool = false
end
name(config::Kmeans_clustering_config) = "K-means"

"""
	clustering(voters::Vector{Abstract_voter}, config::Kmeans_clustering_config, projections=nothing)

Returns the labels and clusters of the voters using the K-means clustering algorithm.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters in the election.
- `config::Kmeans_clustering_config`: The configuration of the clustering algorithm.
- `projections::Matrix{Float64}`: The projections of the voters. If nothing, the projections are calculated from the voters.

# Returns
- `Vector{Int64}`: The labels of the voters.
- `Vector{Vector{Abstract_voter}}`: The clusters of the voters.
"""
function clustering(voters, clustering_config::Kmeans_clustering_config, projections=nothing)
	opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

	best_k = clustering_config.automatic_k ? best_k_silhouettes(opinions, clustering_config.cluster_count) : clustering_config.cluster_count
	kmeans_res = Clustering.kmeans(opinions, best_k)
	labels = kmeans_res.assignments

	clusters = clusterize(labels)

	return labels, clusters
end
