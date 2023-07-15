@kwdef struct DBSCAN_clustering_config <: Abstract_clustering_config
	eps::Float64
	minpts::Int64
end
name(config::DBSCAN_clustering_config) = "DBSCAN"

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