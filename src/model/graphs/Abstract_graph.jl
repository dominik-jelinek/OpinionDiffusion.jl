"""
	init_graph(voters::Vector{Vote}, graph_config::Abstract_graph_config)

Initializes a graph with the given voters and graph_config.

# Arguments
- `voters::Vector{Vote}`: The voters to initialize the graph with.
- `graph_config::Abstract_graph_config`: The config to initialize the graph with.

# Returns
- `g::AbstractGraph`: The initialized graph.
"""
function init_graph(voters::Vector{Vote}, graph_config::Abstract_graph_config)
	throw(NotImplementedError("init_graph"))
end

"""
	weighted_in_degree(g, v, self_loops=false)

Returns the weighted in degree of the given vertex in the given graph.

# Arguments
- `g::AbstractGraph`: The graph to get the weighted in degree from.
- `v::Vertex`: The vertex to get the weighted in degree from.
- `self_loops::Bool=false`: Whether to include self loops in the weighted in degree.

# Returns
- `degree::Int`: The weighted in degree of the given vertex in the given graph.
"""
function weighted_in_degree(g, v, self_loops=false)
	in_edges = inneighbors(g, v)
	degree = 0

	for in_edge in in_edges
		if self_loops || in_edge != v
			degree += get_prop(g, v, in_edge, :weight)
		end
	end

	return degree
end

"""
	ego(social_network, node_id, depth)

Returns the ego network of the given node in the given social_network.

# Arguments
- `social_network::AbstractGraph`: The social network to get the ego network from.
- `node_id::Vertex`: The node to get the ego network from.
- `depth::Int`: The depth of the ego network.

# Returns
- `ego_network::AbstractGraph`: The ego network of the given node in the given social_network.
"""
function ego(social_network, node_id, depth)
	neighs = Graphs.neighbors(social_network, node_id)
	ego_nodes = Set(neighs)
	push!(ego_nodes, node_id)

	front = Set(neighs)
	for i in 1:depth-1
		new_front = Set()
		for voter in front
			union!(new_front, Graphs.neighbors(social_network, voter))
		end
		front = setdiff(new_front, ego_nodes)
		union!(ego_nodes, front)
	end

	return induced_subgraph(social_network, ego_nodes)
end

"""
	drawGraph(g, clustering, K)

Draws the given graph with the given clustering and K.

# Arguments
- `g::AbstractGraph`: The graph to draw.

# Returns
- `nothing`
"""
function drawGraph(g, clustering, K)
	ClusterColor = distinguishable_colors(K)
	nodefillc = ClusterColor[clustering]
	nodesize = [log(Graphs.degree(g, v)) for v in Graphs.vertices(g)]
	GraphPlot.gplot(g,
		nodefillc=nodefillc,
		nodesize=nodesize,
		#edgestrokec=colorant"red",
		layout=GraphPlot.spring_layout)
end
