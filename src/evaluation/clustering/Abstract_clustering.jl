"""
	clustering(voters, clustering_config::Abstract_clustering_config, projections=nothing)

# Arguments
- `voters`: a vector of voters
- `clustering_config`: a clustering configuration
- `projections`: a vector of projections

# Returns
- `clusters`: a vector of tuples (label, set of indices)
"""
function clustering(voters, clustering_config::Abstract_clustering_config, projections=nothing)
	throw(NotImplementedError("clustering"))
end
name(config::Abstract_clustering_config) = "Missing clustering name"

"""
	clusterize(labels::Vector{Int64})

For each unique label creates a set of indices of voters with that label.

# Arguments
- `labels`: a vector of labels

# Returns
- `clusters`: a vector of tuples (label, set of indices)
"""
function clusterize(labels::Vector{Int64})
	unique_labels = sort(unique(labels))

	clusters = [(label, Set(findall(x -> x == label, labels))) for label in unique_labels]

	return clusters
end

"""
	jaccard_similarity(set1::Set{Int64}, set2::Set{Int64})

Calculates the Jaccard similarity between two sets.

# Arguments
- `set1`: a set of indices
- `set2`: a set of indices

# Returns
- `Float64`: the Jaccard similarity between the two sets
"""
function jaccard_similarity(set1::Set{Int64}, set2::Set{Int64})
	intersection_size = length(intersect(set1, set2))
	union_size = length(union(set1, set2))
	return intersection_size / union_size
end

"""
	create_similarity_matrix(template_clusters::Vector{Tuple{Int64,Set{Int64}}}, clusters::Vector{Tuple{Int64,Set{Int64}}})

Creates a similarity matrix between two sets of clusters.

# Arguments
- `template_clusters`: a vector of tuples (label, set of indices)
- `clusters`: a vector of tuples (label, set of indices)

# Returns
- `matrix`: a matrix of similarities between the two sets of clusters
"""
function create_similarity_matrix(template_clusters, clusters)
	matrix = Array{Float64}(undef, length(clusters), length(template_clusters))

	for i in eachindex(clusters)
		for j in eachindex(template_clusters)
			matrix[i, j] = jaccard_similarity(clusters[i][2], template_clusters[j][2])
		end
	end

	return matrix
end

"""
	unify_clusters!(template_clusters::Vector{Tuple{Int64,Set{Int64}}}, clusters::Vector{Tuple{Int64,Set{Int64}}})

Unifies the clusters in `clusters` with the clusters in `template_clusters`.

# Arguments
- `template_clusters`: a vector of tuples (label, set of indices)
- `clusters`: a vector of tuples (label, set of indices)
"""
function unify_clusters!(template_clusters::Vector{Tuple{Int64,Set{Int64}}}, clusters::Vector{Tuple{Int64,Set{Int64}}})
	similarity_matrix = create_similarity_matrix(template_clusters, clusters)
	changed = Vector{Bool}(undef, length(clusters))

	for _ in 1:length(template_clusters)
		max_similarity, max_index = findmax(similarity_matrix)
		if max_similarity == 0.0
			break
		end
		cluster_index, template_index = Tuple(max_index)

		template_id = template_clusters[template_index][1]
		clusters[cluster_index] = (template_id, clusters[cluster_index][2])
		changed[cluster_index] = true

		similarity_matrix[cluster_index, :] .= 0.0
		similarity_matrix[:, template_index] .= 0.0
	end

	max_ID = maximum([cluster[1] for cluster in template_clusters])
	counter = 0
	for i in 1:length(clusters)
		if !changed[i]
			clusters[i] = (max_ID + counter, clusters[i][2])
			counter += 1
		end
	end
end

"""
	best_k_elbow(opinions, max_clusters::Int)

Calculates the best number of clusters using the elbow method.

# Arguments
- `opinions`: a matrix of opinions
- `max_clusters`: the maximum number of clusters to try

# Returns
- `best_k`: the best number of clusters
"""
function best_k_elbow(opinions, max_clusters::Int)
	# Calculate the sum of squared errors (SSE) for different numbers of clusters
	sse = zeros(max_clusters)
	for k in 1:max_clusters
		result = Clustering.kmeans(opinions, k)
		sse[k] = result.totalcost
	end

	# Calculate the elbow point
	best_k = 1
	max_slope = 0
	for k in 2:max_clusters-1
		slope = (sse[k-1] - sse[k+1]) / 2
		if slope > max_slope
			max_slope = slope
			best_k = k
		end
	end

	return best_k
end

"""
	best_k_silhouettes(opinions, max_k::Int)

Calculates the best number of clusters using the silhouette method.

# Arguments
- `opinions`: a matrix of opinions
- `max_k`: the maximum number of clusters to try

# Returns
- `best_k`: the best number of clusters
"""
function best_k_silhouettes(opinions, max_k::Int)
	best_k = 2
	best_silhouette_avg = -1.0
	distances = Distances.pairwise(Distances.Cityblock(), opinions, dims=2)

	for k in 2:max_k
		kmeans_result = Clustering.kmeans(opinions, k)
		assignments = Clustering.assignments(kmeans_result)
		silhouette_vals = Clustering.silhouettes(assignments, distances)
		silhouette_avg = StatsBase.mean(silhouette_vals)

		if silhouette_avg > best_silhouette_avg
			best_silhouette_avg = silhouette_avg
			best_k = k
		end
	end

	return best_k
end

"""
	get_cluster_graph(model, clusters, labels, projections)

Creates a graph of clusters.

# Arguments
- `model`: a model
- `clusters`: a vector of tuples (label, set of indices)
- `labels`: a vector of labels
- `projections`: a matrix of projections

# Returns
- `cluster_graph`: a graph of clusters
"""
function get_cluster_graph(model, clusters, labels, projections)
	g = get_social_network(model)
	voters = get_voters(model)

	cluster_graph = MetaGraph(length(clusters))
	set_prop!(cluster_graph, :ne, ne(g))
	set_prop!(cluster_graph, :nv, nv(g))

	#each vertex has the size equal to coresponding group size
	for (i, (label, indices)) in enumerate(clusters)
		set_prop!(cluster_graph, i, :label, label)
		set_prop!(cluster_graph, i, :indices, indices)
		set_prop!(cluster_graph, i, :pos, StatsBase.mean(projections[:, collect(indices)], dims=2))
	end

	#for each edge in graph add one to grouped graph and if there is one already
	#increase the size of it
	for e in edges(g)
		src_ = findfirst(x -> x[1] == labels[src(e)], clusters)
		dst_ = findfirst(x -> x[1] == labels[dst(e)], clusters)
		edge = Edge(src_, dst_)
		if add_edge!(cluster_graph, edge)
			set_prop!(cluster_graph, edge, :weight, 1)
			set_prop!(cluster_graph, edge, :dist, get_distance(voters[src(e)], voters[dst(e)]))
		else
			weight = get_prop(cluster_graph, edge, :weight)
			set_prop!(cluster_graph, edge, :weight, weight + 1)

			dist = get_prop(cluster_graph, edge, :dist)
			set_prop!(cluster_graph, edge, :dist, dist + get_distance(voters[src(e)], voters[dst(e)]))
		end
	end

	return cluster_graph
end

"""
	cluster_graph_metrics(cluster_graph, g, voters, can_count)

Calculates metrics for a cluster graph.

# Arguments
- `cluster_graph`: a graph of clusters
- `g`: a social network
- `voters`: a matrix of voters
- `can_count`: a matrix of candidates

# Returns
- `vertex_metrics`: a dictionary of vertex metrics
- `edge_metrics`: a dictionary of edge metrics
"""
function cluster_graph_metrics(cluster_graph::AbstractMetaGraph, g, voters, can_count)
	vertex_metrics = Dict{Any,Dict}()
	edge_metrics = Dict{Any,Dict}()

	for v in vertices(cluster_graph)
		indices = collect(get_prop(cluster_graph, v, :indices))
		subgraph = induced_subgraph(g, indices)[1]

		preferences = get_opinion(voters[indices])

		distances = get_distance(preferences)
		println(distances)
		vertex_metrics[v] = Dict(
			:size => length(indices),
			:avg_positions => get_positions(voters[indices], can_count),
			:self_edges => has_edge(cluster_graph, v, v) ? get_prop(cluster_graph, v, v, :weight) : 0,
			:clustering_coefficient => Graphs.global_clustering_coefficient(subgraph),
			:opinion_diameter => maximum(distances),
			:graph_diameter => Graphs.diameter(subgraph),
			:median_opinion_distance => get_median_distance(distances)
		)
	end

	for e in edges(cluster_graph)
		weight = get_prop(cluster_graph, src(e), dst(e), :weight)
		dist = get_prop(cluster_graph, src(e), dst(e), :dist)
		avg_dist = dist / weight

		if src(e) == dst(e)
			edge_ratios = nothing
			homophily = nothing
		else
			if is_directed(cluster_graph)
				edge_ratios = weight / weighted_in_degree(cluster_graph, dst(e))
			else
				to_src = weight / weighted_in_degree(cluster_graph, src(e))
				to_dst = weight / weighted_in_degree(cluster_graph, dst(e))
				edge_ratios = (to_dst, to_src)
			end

			# expected number of edges between two clusters in a random graph.
			random_edges = 2 * length(get_prop(cluster_graph, src(e), :indices)) * length(get_prop(cluster_graph, dst(e), :indices)) / (nv(g) * ne(g))
			# number of edges encountered divided by the expectation in a random graph.
			# if the number is higher than 1, the clusters are more similar than expected.
			# if the number is lower than 1, the clusters are less similar than expected.
			homophily = weight / random_edges
		end

		#proportion of out edges from src to dst, and in edges from dst to src (impact)

		edge_metrics[e] = Dict(
			:weight => weight,
			:dist => dist,
			:avg_dist => avg_dist,
			:edge_ratios => edge_ratios,
			:homophily => homophily
		)
	end

	return vertex_metrics, edge_metrics
end

"""
	draw_cluster_graph!(ax, g)

Draws a cluster graph.

# Arguments
- `ax`: a PyPlot axis
- `g`: a cluster graph
"""
function draw_cluster_graph!(ax, g)
	cluster_labels = [get_prop(g, v, :label) for v in vertices(g)]
	colors = Colors.distinguishable_colors(maximum(cluster_labels))

	nodesizes = [length(get_prop(g, v, :indices)) for v in vertices(g)]
	nodesizes = [nodesize / sum(nodesizes) * 100 for nodesize in nodesizes]
	xs = [get_prop(g, v, :pos)[1, 1] for v in vertices(g)]
	ys = [get_prop(g, v, :pos)[2, 1] for v in vertices(g)]
	edgesizes = [src(e) != dst(e) ? get_prop(g, e, :weight) / (length(get_prop(g, src(e), :indices)) * length(get_prop(g, dst(e), :indices))) * 100 : 0.0 for e in edges(g)]
	#edgesizes = [ src(e) != dst(e) ? round(digits=2, get_prop(g, e, :weight)) : 0 for e in edges(g)]
	#edgesizes = [ edgesize/sum(edgesizes) * 100 for edgesize in edgesizes]
	distances = [src(e) != dst(e) ? round(digits=2, get_prop(g, e, :dist) / get_prop(g, e, :weight)) : "" for e in edges(g)]

	c = colors[cluster_labels]

	edgelabels = [string(val) for val in distances]
	#[round(get_prop(G, :nv) * get_prop(G, e, :weight) / (2*get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)), digits=2) for e in edges(G)]

	graphplot!(ax, g, layout=g -> Point.(zip(xs, ys)), node_color=c, node_size=nodesizes, edge_width=edgesizes)#, elabels=edgelabels)
	#hidedecorations!(ax); hidespines!(ax)
	#ax.aspect = DataAspect()
	#return f
end
