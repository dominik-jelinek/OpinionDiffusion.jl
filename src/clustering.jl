abstract type Abstract_clustering_config <: Config end
@kwdef struct Kmeans_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end
name(config::Kmeans_clustering_config) = "K-means"

function clustering(voters, clustering_config::Kmeans_clustering_config)
    opinions = reduce(hcat, get_opinion(voters))

    kmeans_res = Clustering.kmeans(opinions, clustering_config.cluster_count; maxiter=200)
    labels = kmeans_res.assignments
    
    clusters = clusterize(labels)
    
    return labels, clusters
end

@kwdef struct GM_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end
name(config::GM_clustering_config) = "GM"

function clustering(voters, clustering_config::GM_clustering_config)
    opinions = reduce(hcat, get_opinion(voters))

    data_T = permutedims(opinions)
    gm = ScikitLearn.GaussianMixture(n_components=clustering_config.cluster_count).fit(data_T)
    labels = gm.predict(data_T) .+ 1
    
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
function clustering(voters, clustering_config::Party_clustering_config)
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

function clustering(voters, clustering_config::DBSCAN_clustering_config)
    res = Clustering.dbscan(get_distance(voters), clustering_config.eps, clustering_config.minpts)
    labels = res.assignments .+1
    
    clusters = clusterize(labels)
    
    return labels, clusters
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
function unify_labels!(template_clusters, clusters)
    for i in 1:length(template_clusters)
        best = 0.0
        idx = i
        for j in i:length(clusters)
            similarity = length(intersect(template_clusters[i], clusters[j])) 
            if similarity > best
                best = similarity
                idx = j
            end
        end
        clusters[i], clusters[idx] = clusters[idx], clusters[i]
    end
end


"""
Finds all local maxima in a vector of real values and returns their indices
"""
function find_local_maxima(a)
    max_indices = Vector{Int64}()

    start = 1
    peak = a[1] == a[2] ? true : false
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

    if peak && a[end - 1] == a[end]
        push!(maxima, (start, length(a)))
    end

    if a[end - 1] < a[end]
        push!(maxima, (length(a), length(a)))
    end

    return max_indices
end

Base.@kwdef struct Density_clustering_config <: Abstract_clustering_config
    round_digits::Int64
    projections
end

function clustering(voters, clustering_config::Density_clustering_config)
    projections = clustering_config.projections
    
    result = kde((projections[1, :], projections[2, :]))
    x_start = result.x[1]
    x_step = result.x.step.hi
    y_start = result.y[1]
    y_step = result.y.step.hi

    density_map = round.(result.density; digits=clustering_config.round_digits)
    label_map = watershed(density_map, step_size)
    
    labels = zeros(length(voters))
    for (i, (x_proj, y_proj)) in enumerate(eachcol(projections))
        x_coord = round((x_proj - x_start) / x_step)
        y_coord = round((y_proj - y_start) / y_step)

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
    for (i, col) in enumerate(eachcol(density_map)) peaks_mask_y[i, find_local_maxima(col)] .= true end 

    peaks_mask_x = zeros(Bool, size(density_map, 1), size(density_map, 2))
    for (i, row) in enumerate(eachrow(density_map)) peaks_mask_x[find_local_maxima(row), i] .= true end 

    peaks_mask = peaks_mask_y .& peaks_mask_x
    steps = ceil(maximum(density_map) / step_size)
    queue = Vector{Vector{CartesianIndex}}(undef, steps)
    queue[1] = vec(CartesianIndices((size(density_map, 1), size(density_map, 2))))[vec(peaks_mask)]

    density_map = -density_map
    
    label_depths = [density_map[idx] for idx in queue]
    labels = init_labels(queue)
    open = zeros(Bool, size(density_map, 1), size(density_map, 2))
    
    water_level = step_size
    for step in steps
        expand_basins!(density_map, queue, step, water_level, open, labels, label_depths)
        water_level += step_size
    end 
    
    return labels
end

function expand_basins!(density_map, queue, step, water_level, open, labels, label_depths)
    while !isempty(queue[step])
        idx = popfirst!(queue[step])
        label = labels[idx]
        for N in [CartesianIndex(0, -1), CartesianIndex(-1, 0), CartesianIndex(1, 0), CartesianIndex(0, 1)]
            nei_idx = idx + N
            if !inbounds(density_map, nei_idx) || open[nei_idx] || density_map[idx] > density_map[nei_idx]
                continue
            end
            
            # do not go to already visited points
            if labels[nei_idx] != 0
                # relabel if other label is encountered
                if labels[nei_idx] != label
                    relabel!(labels, label, labels[nei_idx], label_depths)
                end

                continue
            end
            
            # do not go to points that are not reachable with current water level
            if label_depths[label] + water_level < density_map[nei_idx]
                open[nei_idx] = true
                labels[nei_idx] = label

                to_step = (density_map[nei_idx] - label_depths[label] - water_level) / step_size |> ceil |> Int64
                push!(queue[to_step], nei_idx)
                continue
            end

            labels[nei_idx] = label
            push!(queue[step], nei_idx)
        end  
    end
end

function relabel!(label_map, label_1, label_2, label_depths)
    if label_depths[label_1] > label_depths[label_2]
        label_1, label_2 = label_2, label_1
    end

    for (idx, label) in enumerate(label_map)
        if label == label_2
            label_map[idx] = label_1
        end
    end
end

function init_labels(queue)
    labels = zeros(Int64, size(density_map, 1), size(density_map, 2))

    for (i, idx) in enumerate(queue)
        labels[idx] = i
    end

    return labels
end