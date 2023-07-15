@kwdef struct GM_clustering_config <: Abstract_clustering_config
	cluster_count::Int64
	automatic_k::Bool = false
end
name(config::GM_clustering_config) = "GM"

function clustering(voters, clustering_config::GM_clustering_config, projections=nothing)
	opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

	data_T = permutedims(opinions)
	best_k = clustering_config.automatic_k ? best_k_silhouettes(opinions, clustering_config.cluster_count) : clustering_config.cluster_count

	gmm = GaussianMixtures.GMM(best_k, data_T)
	#GaussianMixtures.em!(gmm::GMM, opinions)
	llpg_X = permutedims(GaussianMixtures.llpg(gmm, data_T))
	labels = [m[1] for m in vec(permutedims(argmax(llpg_X, dims=1)))]
	clusters = clusterize(labels)

	return labels, clusters
end