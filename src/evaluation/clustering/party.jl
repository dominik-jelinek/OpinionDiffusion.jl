@kwdef struct Party_clustering_config <: Abstract_clustering_config
	candidates::Vector{Candidate}
end
name(config::Party_clustering_config) = "Party"

"""
	clustering(voters::Vector{Abstract_voter}, config::Party_clustering_config, projections=nothing)

Returns the labels and clusters of the voters using the party clustering algorithm.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters in the election.
- `config::Party_clustering_config`: The configuration of the clustering algorithm.
- `projections::Matrix{Float64}`: The projections of the voters. If nothing, the projections are calculated from the voters.

# Returns
- `Vector{Int64}`: The labels of the voters.
- `Vector{Vector{Abstract_voter}}`: The clusters of the voters.
"""
function clustering(voters, clustering_config::Party_clustering_config, projections=nothing)
	candidates = clustering_config.candidates

	labels = [get_party_ID(candidates[iterate(get_vote(voter)[1])[1]]) for voter in voters]

	clusters = clusterize(labels)

	return labels, clusters
end
