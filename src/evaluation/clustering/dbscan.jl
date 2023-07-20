@kwdef struct DBSCAN_clustering_config <: Abstract_clustering_config
	eps::Float64
	minpts::Int64
end
name(config::DBSCAN_clustering_config) = "DBSCAN"

"""
	clustering(voters::Vector{Abstract_voter}, config::DBSCAN_clustering_config, projections=nothing)

Returns the labels and clusters of the voters using the DBSCAN clustering algorithm.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters in the election.
- `config::DBSCAN_clustering_config`: The configuration of the clustering algorithm.
- `projections::Matrix{Float64}`: The projections of the voters. If nothing, the projections are calculated from the voters.

# Returns
- `Vector{Int64}`: The labels of the voters.
- `Vector{Vector{Abstract_voter}}`: The clusters of the voters.
"""
function clustering(voters, clustering_config::DBSCAN_clustering_config, projections=nothing)
	opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

	res = Clustering.dbscan(opinions, clustering_config.eps; min_neighbors=clustering_config.minpts)
	labels = Vector{Int64}(undef, length(voters))
	for (i, dbscanCluster) in enumerate(res)
		labels[dbscanCluster.core_indices] .= i
		labels[dbscanCluster.boundary_indices] .= i
	end

	clusters = clusterize(labels)

	return labels, clusters
end
