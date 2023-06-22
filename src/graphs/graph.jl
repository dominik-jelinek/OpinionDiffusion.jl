function init_graph(voters, graph_init_config::T) where {T<:Abstract_graph_init_config}
   throw(NotImplementedError("init_graph"))
end

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

   #each vertex has the size equal to coresponding group size
   for (i, (label, indices)) in enumerate(clusters)
      set_prop!(cluster_graph, i, :label, label)
      set_prop!(cluster_graph, i, :indices, indices)
      set_prop!(cluster_graph, i, :pos, Statistics.mean(projections[:, collect(indices)], dims=2))
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