function barabasi_albert_graph(voters::Vector{T}, m::Integer) where T <: Abstract_voter
   if m > length(voters)
      throw(ArgumentError("Argument m for Barabasi-Albert graph creation is higher than number of voters."))
   end

   social_network = Graphs.SimpleGraph(length(voters))
   rand_perm = Random.shuffle(1:length(voters))
   
   degrees = zeros(Float64, length(voters))
   for i in 1:m
      degrees[i] = 1.0
   end

   probs = zeros(Float64, length(voters))
   for i in m+1:length(voters)
      self = rand_perm[i]
      
      #calculate distribution of probabilities for each previously added vertex
      for j in 1:i-1
         probs[j] = degrees[j] * 1.0/(1.0 + get_distance(voters[self], voters[rand_perm[j]]))
      end
      edge_ends = StatsBase.sample(1:length(voters), StatsBase.Weights(probs), m, replace=false)
      
      #add edges
      self = rand_perm[i]
      for edge_end in edge_ends
         Graphs.add_edge!(social_network, self, rand_perm[edge_end]) #probs[j] - 1.0
         degrees[edge_end] += 1
      end
      degrees[i] += 1.0 + m 

   end

   return social_network
end

function init_graph(n::Int, edges::Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}})
   g = Graphs.SimpleGraphFromIterator(edges)
   
   Graphs.add_vertices!(g, n - Graphs.nv(g))
   
   return g
end

# Not used below___________________________________________________________________

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
   ClusterColor  = distinguishable_colors(K)
   nodefillc = ClusterColor[clustering]
   nodesize = [log(Graphs.degree(g, v)) for v in Graphs.vertices(g)]
   GraphPlot.gplot(g,
      nodefillc=nodefillc,
      nodesize=nodesize,
      #edgestrokec=colorant"red",
      layout=GraphPlot.spring_layout)
end

function drawClusteredMetaGraph(G)
   nodesize = [get_prop(G, v, :size) for v in vertices(G)]
   #edgesizes = [get_prop(G, e, :weight) for e in edges(G)]
   edgesizes = [1000*get_prop(G, e, :weight) / (get_prop(G, src(e), :size) * get_prop(G, dst(e), :size)) for e in edges(G)]
   
   display(edgesizes)
   display(GraphPlot.gplot(G,
      nodelabel=1:Graphs.nv(G), 
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
