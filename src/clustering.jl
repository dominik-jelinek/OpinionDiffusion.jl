abstract type Abstract_clustering_config <: Config end
@kwdef struct Kmeans_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end
name(config::Kmeans_clustering_config) = "K-means"

function clustering(voters, clustering_config::Kmeans_clustering_config, projections=nothing)
    opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

    kmeans_res = Clustering.kmeans(opinions, clustering_config.cluster_count)
    labels = kmeans_res.assignments
    
    clusters = clusterize(labels)
    
    return labels, clusters
end

@kwdef struct GM_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end
name(config::GM_clustering_config) = "GM"

function clustering(voters, clustering_config::GM_clustering_config, projections=nothing)
    opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

    data_T = permutedims(opinions)
    gmm = GaussianMixtures.GMM(clustering_config.cluster_count, data_T)
    #GaussianMixtures.em!(gmm::GMM, opinions)
    llpg_X = permutedims(GaussianMixtures.llpg(gmm, data_T))
    labels = [ m[1] for m in vec(permutedims(argmax(llpg_X, dims=1)))]
    clusters = clusterize(labels)
    
    return labels, clusters
end

@kwdef struct Party_clustering_config <: Abstract_clustering_config
    candidates::Vector{Candidate}
end
name(config::Party_clustering_config) = "Party"

"""
Cluster voters based on highest ranked candidate
"""
function clustering(voters, clustering_config::Party_clustering_config, projections=nothing)
    candidates = clustering_config.candidates

    labels = [candidates[iterate(get_vote(voter)[1])[1]].party for voter in voters] 
    
    clusters = clusterize(labels)
    
    return labels, clusters
end

@kwdef struct DBSCAN_clustering_config <: Abstract_clustering_config
    eps::Float64
    minpts::Int64
end
name(config::DBSCAN_clustering_config) = "DBSCAN"

function clustering(voters, clustering_config::DBSCAN_clustering_config, projections=nothing)
    res = Clustering.dbscan(get_distance(projections === nothing ? voters : projections), clustering_config.eps, clustering_config.minpts)
    labels = res.assignments .+1
    
    clusters = clusterize(labels)
    
    return labels, clusters
end

Base.@kwdef struct Density_clustering_config <: Abstract_clustering_config
    round_digits::Int64
end
name(config::Density_clustering_config) = "Kernel Density clustering"

function clustering(voters, clustering_config::Density_clustering_config, projections)
    @assert(size(projections, 1) == 2)

    result = KernelDensity.kde((projections[1, :], projections[2, :]))
    x_start = result.x[1]
    x_step = result.x.step.hi
    y_start = result.y[1]
    y_step = result.y.step.hi

    density_map = round.(result.density; digits=clustering_config.round_digits)
    step_size = maximum(density_map) / (10 * clustering_config.round_digits)
    
    density_map = -density_map
    label_map, merges = watershed(density_map, step_size)
    #push!(ploty, Makie.heatmap(label_map, transparency=true, colormap=:viridis, colorrange=(0, maximum(label_map))))

    labels = zeros(Int64, length(voters))
    for (i, (x_proj, y_proj)) in enumerate(eachcol(projections))
        x_coord = round((x_proj - x_start) / x_step) |> Int64
        y_coord = round((y_proj - y_start) / y_step) |> Int64

        labels[i] = label_map[y_coord, x_coord]
    end

    clusters = clusterize(labels)
    
    return labels, clusters
end

"""
Finds all basins and then uses watershed algorithm adaptation to expand them in order to define areas that belong to the clusters.
"""
function watershed(density_map, step_size)
    peaks_mask_y = zeros(Bool, size(density_map, 1), size(density_map, 2))
    for (i, col) in enumerate(eachcol(density_map)) peaks_mask_y[find_local_minima(col), i] .= true end 

    peaks_mask_x = zeros(Bool, size(density_map, 1), size(density_map, 2))
    for (i, row) in enumerate(eachrow(density_map)) peaks_mask_x[i, find_local_minima(row)] .= true end 

    peaks_mask = peaks_mask_y .& peaks_mask_x
    steps = abs(minimum(density_map)) / step_size |> ceil |> Int64

    peaks = vec(CartesianIndices((size(density_map, 1), size(density_map, 2))))[vec(peaks_mask)]

    label_depths = [density_map[idx] for idx in peaks]
    depth = density_map[peaks[argmin(label_depths)]]

    queue = [Vector{CartesianIndex}() for _ in 1:steps+1]
    for peak in peaks
        push!(queue[((density_map[peak] - depth) / step_size |> ceil |> Int64) + 1], peak)
    end
    
    label_map = init_labels(peaks, (size(density_map, 1), size(density_map, 2)))
    
    water_level = step_size
    merges = Dict()
    ploty = Vector{Any}()
    for step in eachindex(queue)
        push!(ploty, Makie.heatmap(label_map, transparency=true, colormap=:viridis, colorrange=(0, maximum(label_map))))
        expand_basins!(density_map, queue, step, step_size, depth, water_level, label_map, label_depths, merges)
        water_level += step_size
    end 
    for label in keys(merges)
        step, indices = merges[label]
        label_map[indices] .= label 
    end
    #push!(ploty, Makie.heatmap(label_map, transparency=true, colormap=:viridis, colorrange=(0, maximum(label_map))))

    return label_map, merges#, ploty
end

function inbounds(matrix, coord)
    return !(coord[1] < 1 || coord[2] < 1 || coord[1] > size(matrix, 1) || coord[2] > size(matrix, 2))
end

function expand_basins!(density_map, queue, step, step_size, depth, water_level, label_map, label_depths, merges)
    while !isempty(queue[step])
        idx = popfirst!(queue[step])
        for N in [CartesianIndex(0, -1), CartesianIndex(-1, 0), CartesianIndex(1, 0), CartesianIndex(0, 1)]
            nei_idx = idx + N
            if !inbounds(density_map, nei_idx) || density_map[idx] > density_map[nei_idx] #|| density_map[nei_idx] == 0
                continue
            end
            
            # do not go to already visited points
            if label_map[nei_idx] != 0
                # relabel if other label is encountered
                if label_map[nei_idx] != label_map[idx]
                    merged_label, indices = relabel!(label_map, label_map[idx], label_map[nei_idx], label_depths, depth+water_level)
                    merges[merged_label] = (step, indices)
                end

                continue
            end
            
            # Do not go to points that are not reachable with current water level
            # Instead label them with current label and add to queue for later
            if depth + water_level < density_map[nei_idx]
                label_map[nei_idx] = label_map[idx]

                to_step = ((density_map[nei_idx] - depth) / step_size |> ceil |> Int64) + 1
                push!(queue[to_step], nei_idx)
                continue
            end

            label_map[nei_idx] = label_map[idx]
            push!(queue[step], nei_idx)
        end  
    end

    return merges
end

function relabel!(label_map, label_1, label_2, label_depths, point_depth)
    if label_depths[label_1] > label_depths[label_2]
        label_1, label_2 = label_2, label_1
    end
    # label_1 is deeper than label_2

    merged_pos = [] 
    for (idx, label) in enumerate(label_map)
        if label == label_2
            push!(merged_pos, idx)
            #if abs(point_depth - label_depths[label_2]) < 1/5 * abs(point_depth - label_depths[label_1])
            #    label_map[idx] = label_1
            #end
        end
    end

    return label_2, merged_pos
end

function init_labels(queue, shape)
    labels = zeros(Int64, shape)

    for (i, idx) in enumerate(queue)
        labels[idx] = i
    end

    return labels
end

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
    
    clusters = [(label, Set(findall(x->x==label, labels))) for label in unique_labels]
   
    return clusters
end

""" TODO
Finds the best bijection from clusters to template_clusters based on set overlap
"""
function unify_clusters!(template_clusters, clusters, old_projections , projections)
    jaccard_dist_sim = zeros(length(template_clusters), length(clusters))
    mean_distance = 0
    for i in eachindex(template_clusters)
        for j in eachindex(sorted)
            jaccard_sim = length(intersect(template_clusters[i][2], clusters[j][2])) / length(union(template_clusters[i][2], clusters[j][2]))
            distance = get_avg_distance(old_projections[:, collect(template_clusters[i][2])], projections[:, collect(clusters[j][2])])
            mean_distance += distance

            jaccard_dist_sim[i, j] = (1.0 - jaccard_sim) * distance
        end
    end
    mean_distance = mean_distance / (length(template_clusters) * length(clusters))

    # find best bijection until jaccard similarity is less than 0.5 * mean_distance
    new_labels = zeros(length(clusters))
    while true
        val, min_idx = findmin(jaccard_dist_sim)
        if val > 0.4 * mean_distance
            break
        end

        new_labels[min_idx[2]] = template_clusters[min_idx[1]][1]
        jaccard_dist_sim[min_idx[1], :] .= Inf
        jaccard_dist_sim[:, min_idx[2]] .= Inf
    end

    # create new labels for clusters that were not matched
    max_label = maximum([label for (label, _) in template_clusters])
    counter = 1
    for i in eachindex(new_labels)
        if new_labels[i] == 0
            new_labels[i] = max_label + counter
            counter += 1
        end
    end

    # update clusters
    for i in eachindex(clusters)
        clusters[i] = (new_labels[i], clusters[i][2])
    end
end

find_local_minima(a) = find_local_maxima(-a)
"""
Finds all local maxima in a vector of real values and returns their indices
"""
function find_local_maxima(a)
    max_indices = Vector{Int64}()

    start = 1
    peak = a[1] == a[2]
    if a[1] > a[2]
        push!(max_indices, 1)
    end

    for i in 2:length(a) - 1
        # test inflection point
        if peak && a[i] < a[i + 1]
            peak = false
        end

        # test start of the peak
        if a[i - 1] < a[i] && a[i] >= a[i + 1]
            start = i
            peak = true
        end

        # test end of the peak
        if peak && a[i] > a[i + 1]
            append!(max_indices, collect(start:i))
            peak = false
        end
    end

    if peak && a[end - 1] == a[end] && start != 1
        append!(max_indices, collect(start:length(a)))
    end

    if a[end - 1] < a[end]
        push!(max_indices, length(a))
    end

    return max_indices
end