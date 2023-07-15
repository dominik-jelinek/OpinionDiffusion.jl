@kwdef struct Party_clustering_config <: Abstract_clustering_config
	candidates::Vector{Candidate}
end
name(config::Party_clustering_config) = "Party"

"""
Cluster voters based on highest ranked candidate
"""
function clustering(voters, clustering_config::Party_clustering_config, projections=nothing)
	candidates = clustering_config.candidates

	labels = [get_party_ID(candidates[iterate(get_vote(voter)[1])[1]]) for voter in voters]

	clusters = clusterize(labels)

	return labels, clusters
end