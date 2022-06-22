function generate_edges(voters::Vector{T}, dist_metric::Distances.Metric, edge_init_func::Function) where T <: Abstract_voter
   edges = Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}}()

   @inbounds for i in 1:(length(voters) - 1)

      @inbounds for j in (i + 1):length(voters)
         if rand() <= edge_init_func(Distances.evaluate(dist_metric, voters[i].opinion, voters[j].opinion))
            push!(edges, Graphs.SimpleGraphs.SimpleEdge{Int64}(i, j))
         end
      end
   end

   return edges
end

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

function weighted_barabasi_albert_graph(voters::Vector{T}, m::Integer, ratio=1.0) where T <: Abstract_voter
   if m > length(voters)
      throw(ArgumentError("Argument m for Barabasi-Albert graph creation is higher than number of voters."))
   end
   n = length(voters)
   social_network = MetaGraphs.MetaGraph(n)

   rand_perm = Random.shuffle(1:n)
   voters_perm = voters[rand_perm]

   # add first node that is connected to all other nodes
   @inbounds for i in 1:m
      add_edge!(social_network, rand_perm[m + 1], rand_perm[i])
      set_prop!(social_network, rand_perm[m + 1], rand_perm[i], :weight, get_distance(voters_perm[m + 1], voters_perm[i]))
   end
   degrees = zeros(Float64, n)
   for i in 1:m
      degrees[i] = 1.0
   end
   degrees[m + 1] += m
   degree_sum = m

   # add the rest of nodes and generate edges based on opinion similarity and popularity of other votes
   probs = zeros(Float64, n)
   for i in m+2:length(voters_perm)
      self = voters_perm[i]
      
      distances = 1.0 ./ (1.0 .+ get_distance(self, voters_perm[1:i-1]))
      #calculate distribution of p robabilities for each previously added vertex
      dist_sum = sum(distances)
      @inbounds for j in eachindex(distances)
         probs[j] = ratio * degrees[j] / degree_sum + (1.0 - ratio) * distances[j] / dist_sum
      end
      edge_ends = StatsBase.sample(1:n, StatsBase.Weights(probs), m, replace=false)
      
      #add edges
      @inbounds for edge_end in edge_ends
         add_edge!(social_network, rand_perm[i], rand_perm[edge_end])
         set_prop!(social_network, rand_perm[i], rand_perm[edge_end], :weight, 1.0)#distances[edge_end])
         degrees[edge_end] += 1.0
      end

      degrees[i] += m 
      degree_sum += 2 * m
   end

   return social_network
end

function get_DEG(voters, targed_deg_distr, target_cc; ratio=1.0, log_lvl=true)
   n = length(voters)
   social_network = MetaGraphs.MetaGraph(n)

   rand_perm = Random.shuffle(1:n)
   voters_perm = voters[rand_perm]

   m = floor(sum(targed_deg_distr) / 2)
   rds = deepcopy(targed_deg_distr[rand_perm])
   T = floor(target_cc * sum([ choose2(rd) for rd in rds]) / 3)
   println("m: ", m, " T: ", T)
   limit = 0
   while T > 0
      limit += 1
      if limit == 1000
         break
      end
      
      probs = rds
      if log_lvl
         println(T)
         println(StatsBase.Weights(probs))
      end
      u = StatsBase.sample(1:n, StatsBase.Weights(probs))

      distances_u = 1.0 ./ (1.0 .+ get_distance(voters_perm[u], voters_perm))
      probs = ratio .* rds ./ sum(rds) .+ (1.0 - ratio) .* distances_u ./ sum(distances_u)
      probs = probs .* (rds.>0)
      probs[u] = 0.0
      if log_lvl
         println(probs)
      end
      v = StatsBase.sample(1:n, StatsBase.Weights(probs))

      distances_v = 1.0 ./ (1.0 .+ get_distance(voters_perm[v], voters_perm))
      probs = ratio .* rds ./ sum(rds) .+ (1.0 - ratio) .* (distances_u ./ sum(distances_u) .+ distances_v ./ sum(distances_v)) ./ 2 
      probs = probs .* (rds.>0)
      probs[u] = 0.0
      probs[v] = 0.0
      
      w = StatsBase.sample(1:n, StatsBase.Weights(probs))
      if log_lvl
         println(probs)
         println(u, v, w)
      end
      
      # test feasibility of creating a triangle with vertices u, v, w
      if rds[u] == 1 && !has_edge(social_network, u, v) && !has_edge(social_network, u, w)
         #T -= 1
         continue
      end

      if rds[v] == 1 && !has_edge(social_network, v, u) && !has_edge(social_network, v, w)
         #T -= 1
         continue
      end

      if rds[w] == 1 && !has_edge(social_network, w, u) && !has_edge(social_network, w, v)
         #T -= 1
         continue
      end

      edges_added = 0
      for (v_1, v_2) in zip([u, v, w], [v, w, u])
         if add_edge!(social_network, rand_perm[v_1], rand_perm[v_2])
            edges_added += 1
            rds[v_1] -= 1
            rds[v_2] -= 1
         end
      end
      
      if edges_added > 0   
         m -= edges_added
      end

      # we decrease T even when no new triangles were created as is written in the paper
      T -= 1
   end

   limit = 0
   while m > 0
      if log_lvl
         println(m)
      end

      probs = rds
      u = StatsBase.sample(1:n, StatsBase.Weights(probs))

      distances_u = 1.0 ./ (1.0 .+ get_distance(voters_perm[u], voters_perm))
      probs = ratio .* rds ./ sum(rds) .+ (1.0 - ratio) .* distances_u ./ sum(distances_u)
      probs = probs .* (rds.>1)
      probs[u] = 0.0
      v = StatsBase.sample(1:n, StatsBase.Weights(probs))

      if add_edge!(social_network, rand_perm[u], rand_perm[v])
         rds[u] -= 1
         rds[v] -= 1
         m -= 1
      end

      limit += 1
      if limit == 100
         break
      end
   end

   return social_network
end

function init_graph(vert_count::Int, edges::Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}})
   g = Graphs.SimpleGraphFromIterator(edges)
   
   Graphs.add_vertices!(g, vert_count - Graphs.nv(g))
   
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
   ClusterColor  = Colors.distinguishable_colors(K)
   nodefillc = ClusterColor[clustering]
   display(GraphPlot.gplot(g,
      nodefillc=nodefillc,
      layout=random_layout))
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

#__________________________________________________________________
