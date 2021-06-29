function generate_edges(voters::Vector{T}, dist_metric::Metric, edge_init_func::Function) where T <: Abstract_voter
   edges = Vector{LightGraphs.SimpleGraphs.SimpleEdge{Int64}}()

   @inbounds for i in 1:(length(voters) - 1)

      @inbounds for j in (i + 1):length(voters)
         if rand() <= edge_init_func(Distances.evaluate(dist_metric, voters[i].opinion, voters[j].opinion))
            push!(edges, LightGraphs.SimpleGraphs.SimpleEdge{Int64}(i, j))
         end
      end
   end

   return edges
end

function init_graph(voters::Vector{T}, edge_limit, dist_metric::Metric, edge_init_func::Function) where T <: Abstract_voter
   social_network = SimpleGraph(length(voters))

   edge_counter = 0
   while edge_counter != edge_limit
      self = rand(voters)
      neighbors_ = neighbors(social_network, self.ID)

      #singleton
      if length(neighbors_) == 0
         voter = rand(voters)
         while self.ID == voter.ID 
            voter = rand(voters)
         end

         if rand() <= edge_init_func(Distances.evaluate(dist_metric, self.opinion, voter.opinion))
            if add_edge!(social_network, self.ID, voter.ID)
               edge_counter += 1
            end
         end

         continue
      end

      #pick random neighbor
      neighbor_ID = rand(neighbors_)
      mutual_neighbors = neighbors(social_network, neighbor_ID)

      #path of length one
      if length(mutual_neighbors) == 1
         voter = rand(voters)
         while self.ID == voter.ID 
            voter = rand(voters)
         end

         if rand() <= edge_init_func(Distances.evaluate(dist_metric, self.opinion, voter.opinion))
            if add_edge!(social_network, self.ID, voter.ID) 
               edge_counter += 1
            end
         end

         continue
      end

      #general case
      mutual_neighbor_ID = rand(mutual_neighbors)
      while self.ID == mutual_neighbor_ID
         mutual_neighbor_ID = rand(mutual_neighbors)
      end

      mutual_neighbor = voters[mutual_neighbor_ID]

      if rand() <= edge_init_func(Distances.evaluate(dist_metric, self.opinion, mutual_neighbor.opinion))
         if add_edge!(social_network, self.ID, mutual_neighbor.ID) 
            edge_counter += 1
         end
      end
   end

   return social_network
end

function init_graph(vert_count::Int, edges::Vector{LightGraphs.SimpleGraphs.SimpleEdge{Int64}})
   g = SimpleGraphFromIterator(edges)
   
   add_vertices!(g, vert_count - nv(g))
   
   return g
end

#___________________________________________________________________

function createClusteredMetaGraph(g, clusters, labels)
   G = MetaGraph(length(clusters))
   #each vertex has the size equal to coresponding group size
   for i in 1:length(clusters)
      set_prop!(G, i, :size, length(clusters[i]))
   end
   #for each edge in graph add one to grouped graph and if there is one already
   #increase the size of it
   for e in edges(g)
      if add_edge!(G, labels[src(e)], labels[dst(e)])
         set_prop!(G, labels[src(e)], labels[dst(e)], :weight, 1)
      else
         if labels[src(e)] != labels[dst(e)] #prohibit self edges
            weight = get_prop(G, labels[src(e)], labels[dst(e)], :weight)
            set_prop!(G, labels[src(e)], labels[dst(e)], :weight, weight + 1)
         end
      end
   end
   return G
end

function drawGraph(g, clustering, K)
   ClusterColor  = distinguishable_colors(K, colorant"blue")
   nodefillc = ClusterColor[clustering]
   display(gplot(g,
      nodefillc=nodefillc,
      layout=random_layout))
end

function drawClusteredMetaGraph(G)
   nodesize = [get_prop(G, v, :size) for v in vertices(G)]
   #edgesizes = [get_prop(G, e, :weight) for e in edges(G)]
   edgesizes = [1000*get_prop(G, e, :weight) / (get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)) for e in edges(G)]
   
   display(edgesizes)
   display(gplot(G,
      nodelabel=1:nv(G), 
      nodesize=nodesize,
      edgelinewidth=edgesizes,
      layout=circular_layout))
end

#__________________________________________________________________

function initGraphCompr(database, p, minFriends, maxFriends) #WIP
   g = SimpleGraph(length(database))
   totalDistance = 0

   lowerFriendProb = minFriends/length(database)
   upperFriendProb = maxFriends/length(database)

   candidateBuffer = zeros(Int32, length(database[1]))
   comprDBpref, comprDBgrp = compressDB(database)
   for i in 1:length(comprDBpref)-1
      #based on Kendall tau value assign minFriends to maxFriends edges
      for j in i+1:length(comprDBpref)
         distance = kendallTau!(comprDBpref[i],comprDBpref[j], p, candidateBuffer)
         prob = translateRange(0.0, 1.0, lowerFriendProb, upperFriendProb, 1.0 - distance)
         totalDistance += length(comprDBgrp[i])*length(comprDBgrp[j])*distance

         #edgesToAdd = ceil(Int, prob*length(comprDBgrp[i])*length(comprDBgrp[j]))
         #x = rand(1:length(comprDBgrp[i]), edgesToAdd)
         #y = rand(1:length(comprDBgrp[j]), edgesToAdd)
         #for k in 1:length(x)
         #   add_edge!(g,comprDBgrp[i][x[k]],comprDBgrp[j][y[k]])
         #end

         for k in 1:length(comprDBgrp[i])
            for l in 1:length(comprDBgrp[j])
               if rand() <= prob
                  add_edge!(g,comprDBgrp[i][k],comprDBgrp[j][l])
               end
            end
         end

      end
   end
   return g, totalDistance
end

function initGraphCompr(database, p::Float64, normalize::Bool, f::Function) #WIP
   g = SimpleGraph(length(database))
   totalDistance = 0.0

   candidateBuffer = zeros(Int32, length(database[1]))
   comprDBpref, comprDBgrp = compressDB(database)
   for i in 1:length(comprDBpref)-1
      #based on Kendall tau value assign minFriends to maxFriends edges
      for j in i+1:length(comprDBpref)
         distance = kendallTau!(comprDBpref[i],comprDBpref[j], p, candidateBuffer, normalize)
         totalDistance += length(comprDBgrp[i])*length(comprDBgrp[j])*distance

         for k in 1:length(comprDBgrp[i])
            for l in 1:length(comprDBgrp[j])
               if rand() <= f(distance)
                  add_edge!(g,comprDBgrp[i][k],comprDBgrp[j][l])
               end
            end
         end

      end
   end
   return g, totalDistance/pairsCount(length(database))
end

#__________________________________________________________________
