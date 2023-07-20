@kwdef struct GM_clustering_config <: Abstract_clustering_config
	cluster_count::Int64
	automatic_k::Bool = false
end
name(config::GM_clustering_config) = "GM"

"""
	clustering(voters::Vector{Abstract_voter}, config::GM_clustering_config, projections=nothing)

Returns the labels and clusters of the voters using the Gaussian Mixture clustering algorithm.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters in the election.
- `config::GM_clustering_config`: The configuration of the clustering algorithm.
- `projections::Matrix{Float64}`: The projections of the voters. If nothing, the projections are calculated from the voters.

# Returns
- `Vector{Int64}`: The labels of the voters.
- `Vector{Vector{Abstract_voter}}`: The clusters of the voters.
"""
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
