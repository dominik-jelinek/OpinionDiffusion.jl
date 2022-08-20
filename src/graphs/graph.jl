function init_graph(n::Int, edges::Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}})
   g = Graphs.SimpleGraphFromIterator(edges)
   
   Graphs.add_vertices!(g, n - Graphs.nv(g))
   
   return g
end

function get_cluster_graph(model, clusters, labels, projections)
   g = get_social_network(model)
   voters = get_voters(model)

   cluster_graph = MetaGraph(length(clusters))
   set_prop!(cluster_graph, :ne, ne(g))
   set_prop!(cluster_graph, :nv, nv(g))

   cluster_colors  = Colors.distinguishable_colors(length(clusters))
   #each vertex has the size equal to coresponding group size
   for (i, cluster) in enumerate(clusters)
      set_prop!(cluster_graph, i, :size, length(cluster))
      set_prop!(cluster_graph, i, :color, cluster_colors[i])
      if length(cluster) > 1
         set_prop!(cluster_graph, i, :pos, Statistics.mean(projections[:, collect(cluster)], dims=2))
      else
         set_prop!(cluster_graph, i, :pos, [0,0])
      end
   end

   #for each edge in graph add one to grouped graph and if there is one already
   #increase the size of it
   for e in edges(g)
      if add_edge!(cluster_graph, labels[src(e)], labels[dst(e)])
         set_prop!(cluster_graph, labels[src(e)], labels[dst(e)], :weight, 1)
         set_prop!(cluster_graph, labels[src(e)], labels[dst(e)], :dist, get_distance(voters[src(e)], voters[dst(e)]))
      else
         weight = get_prop(cluster_graph, labels[src(e)], labels[dst(e)], :weight)
         set_prop!(cluster_graph, labels[src(e)], labels[dst(e)], :weight, weight + 1)

         dist = get_prop(cluster_graph, labels[src(e)], labels[dst(e)], :dist)
         set_prop!(cluster_graph, labels[src(e)], labels[dst(e)], :dist, dist + get_distance(voters[src(e)], voters[dst(e)]))
      end
   end

   for e in edges(cluster_graph)
      weight = get_prop(cluster_graph, src(e), dst(e), :weight)
      dist = get_prop(cluster_graph, src(e), dst(e), :dist)
      set_prop!(cluster_graph, src(e), dst(e), :dist, dist / weight)
   end

   return cluster_graph
end

function draw_cluster_graph(g)
   nodesize = [get_prop(g, v, :size) for v in vertices(g)]
   xs = [get_prop(g, v, :pos)[1, 1] for v in vertices(g)]
   ys = [-get_prop(g, v, :pos)[2, 1] for v in vertices(g)]
   #edgesizes = [get_prop(G, e, :weight) / (get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)) for e in edges(G)]
   edgesizes = [ src(e) != dst(e) ? round(digits=2, get_prop(g, e, :weight)) : 0 for e in edges(g)]
   distances = [ src(e) != dst(e) ? round(digits=2, get_prop(g, e, :dist)) : "" for e in edges(g)]

   c = [get_prop(g, v, :color) for v in vertices(g)]
   #=edgesizes = []
   for e in edges(G)
      src_self = has_prop(G, src(e), :self_edges) ? get_prop(G, src(e), :self_edges) : 0
      dst_self = has_prop(G, dst(e), :self_edges) ? get_prop(G, dst(e), :self_edges) : 0
      expected_edges = (get_prop(G, src(e), dst(e), :weight) + src_self + dst_self) * (2*get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)) / (get_prop(G, src(e), :size) + get_prop(G, dst(e), :size))
      ratio = get_prop(G, src(e), dst(e), :weight) / expected_edges
      push!(edgesizes, round(ratio, digits=2))
   end=#
   edgelabels = distances
   #[round(get_prop(G, :nv) * get_prop(G, e, :weight) / (2*get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)), digits=2) for e in edges(G)]

   return GraphPlot.gplot(g, xs, ys, 
                           nodesize=nodesize,
                           #nodelabel=1:Graphs.nv(G), 
                           edgelinewidth=edgesizes,
                           nodefillc=c,
                           edgelabel=edgelabels)
end

function ego(social_network, node_id, depth)
   neighs = Graphs.neighbors(social_network, node_id)
   ego_nodes = Set(neighs)
   push!(ego_nodes, node_id)
   
   front = Set(neighs)
   for i in 1:depth - 1
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
   ClusterColor  = distinguishable_colors(K)
   nodefillc = ClusterColor[clustering]
   nodesize = [log(Graphs.degree(g, v)) for v in Graphs.vertices(g)]
   GraphPlot.gplot(g,
      nodefillc=nodefillc,
      nodesize=nodesize,
      #edgestrokec=colorant"red",
      layout=GraphPlot.spring_layout)
end