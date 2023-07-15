function init_graph(voters::Vector{Vote}, graph_config::Abstract_graph_config)
	throw(NotImplementedError("init_graph"))
end

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