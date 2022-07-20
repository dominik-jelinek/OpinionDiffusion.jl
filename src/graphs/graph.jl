function init_graph(n::Int, edges::Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}})
   g = Graphs.SimpleGraphFromIterator(edges)
   
   Graphs.add_vertices!(g, n - Graphs.nv(g))
   
   return g
end

function get_cluster_graph(g, clusters, labels, projections)
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
      if labels[src(e)] == labels[dst(e)]
         if has_prop(cluster_graph, :self_edges)
            self_edges = get_prop(cluster_graph, labels[src(e)], :self_edges)
            set_prop!(cluster_graph, labels[src(e)], :self_edges, self_edges + 1)
         else
            set_prop!(cluster_graph, labels[src(e)], :self_edges, 1)
         end
         continue
      end

      if add_edge!(cluster_graph, labels[src(e)], labels[dst(e)])
         set_prop!(cluster_graph, labels[src(e)], labels[dst(e)], :weight, 1)
      else
         weight = get_prop(cluster_graph, labels[src(e)], labels[dst(e)], :weight)
         set_prop!(cluster_graph, labels[src(e)], labels[dst(e)], :weight, weight + 1)
      end
   end

   return cluster_graph
end

function draw_cluster_graph(g)
   nodesize = [get_prop(g, v, :size) for v in vertices(g)]
   xs = [get_prop(g, v, :pos)[1, 1] for v in vertices(g)]
   ys = [-get_prop(g, v, :pos)[2, 1] for v in vertices(g)]
   #edgesizes = [get_prop(G, e, :weight) / (get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)) for e in edges(G)]
   edgesizes = [get_prop(g, e, :weight) for e in edges(g)]
   c = [get_prop(g, v, :color) for v in vertices(g)]
   #=edgesizes = []
   for e in edges(G)
      src_self = has_prop(G, src(e), :self_edges) ? get_prop(G, src(e), :self_edges) : 0
      dst_self = has_prop(G, dst(e), :self_edges) ? get_prop(G, dst(e), :self_edges) : 0
      expected_edges = (get_prop(G, src(e), dst(e), :weight) + src_self + dst_self) * (2*get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)) / (get_prop(G, src(e), :size) + get_prop(G, dst(e), :size))
      ratio = get_prop(G, src(e), dst(e), :weight) / expected_edges
      push!(edgesizes, round(ratio, digits=2))
   end=#
   edgelabels = edgesizes
   #[round(get_prop(G, :nv) * get_prop(G, e, :weight) / (2*get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)), digits=2) for e in edges(G)]

   return GraphPlot.gplot(g, xs, ys, 
                           nodesize=nodesize,
                           #nodelabel=1:Graphs.nv(G), 
                           edgelinewidth=edgesizes,
                           nodefillc=c,
                           edgelabel=edgelabels)
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