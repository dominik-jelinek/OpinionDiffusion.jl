@kwdef struct Kmeans_clustering_config <: Abstract_clustering_config
    cluster_count::Int64
end
name(config::Kmeans_clustering_config) = "K-means"

function clustering(voters, clustering_config::Kmeans_clustering_config, projections=nothing)
    opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

    best_k = best_k_silhouettes(opinions, clustering_config.cluster_count)
    kmeans_res = Clustering.kmeans(opinions, best_k)
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
    best_k = best_k_silhouettes(opinions, clustering_config.cluster_count)

    gmm = GaussianMixtures.GMM(best_k, data_T)
    #GaussianMixtures.em!(gmm::GMM, opinions)
    llpg_X = permutedims(GaussianMixtures.llpg(gmm, data_T))
    labels = [m[1] for m in vec(permutedims(argmax(llpg_X, dims=1)))]
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
    opinions = projections === nothing ? reduce(hcat, get_opinion(voters)) : projections

    res = Clustering.dbscan(opinions, clustering_config.eps; min_neighbors=clustering_config.minpts)
    labels = Vector{Int64}(undef, length(voters))
    for (i, dbscanCluster) in enumerate(res)
        labels[dbscanCluster.core_indices] .= i
        labels[dbscanCluster.boundary_indices] .= i
    end

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
    for (i, col) in enumerate(eachcol(density_map))
        peaks_mask_y[find_local_minima(col), i] .= true
    end

    peaks_mask_x = zeros(Bool, size(density_map, 1), size(density_map, 2))
    for (i, row) in enumerate(eachrow(density_map))
        peaks_mask_x[i, find_local_minima(row)] .= true
    end

    peaks_mask = peaks_mask_y .& peaks_mask_x
    steps = abs(minimum(density_map)) / step_size |> ceil |> Int64

    peaks = vec(CartesianIndices((size(density_map, 1), size(density_map, 2))))[vec(peaks_mask)]

    label_depths = [density_map[idx] for idx in peaks]
    depth = density_map[peaks[argmin(label_depths)]]

    queue = [Vector{CartesianIndex}() for _ in 1:steps+1]
    for peak in peaks
        push!(queue[((density_map[peak]-depth)/step_size|>ceil|>Int64)+1], peak)
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
                    merged_label, indices = relabel!(label_map, label_map[idx], label_map[nei_idx], label_depths, depth + water_level)
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

    for i in 2:length(a)-1
        # test inflection point
        if peak && a[i] < a[i+1]
            peak = false
        end

        # test start of the peak
        if a[i-1] < a[i] && a[i] >= a[i+1]
            start = i
            peak = true
        end

        # test end of the peak
        if peak && a[i] > a[i+1]
            append!(max_indices, collect(start:i))
            peak = false
        end
    end

    if peak && a[end-1] == a[end] && start != 1
        append!(max_indices, collect(start:length(a)))
    end

    if a[end-1] < a[end]
        push!(max_indices, length(a))
    end

    return max_indices
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

    clusters = [(label, Set(findall(x -> x == label, labels))) for label in unique_labels]

    return clusters
end

function jaccard_similarity(set1::Set{Int64}, set2::Set{Int64})
    intersection_size = length(intersect(set1, set2))
    union_size = length(union(set1, set2))
    return intersection_size / union_size
end

function create_similarity_matrix(template_clusters, clusters)
    matrix = Array{Float64}(undef, length(clusters), length(template_clusters))

    for i in eachindex(clusters)
        for j in eachindex(template_clusters)
            matrix[i, j] = jaccard_similarity(clusters[i][2], template_clusters[j][2])
        end
    end

    return matrix
end

function unify_clusters!(template_clusters::Vector{Tuple{Int64,Set{Int64}}}, clusters::Vector{Tuple{Int64,Set{Int64}}})
    similarity_matrix = create_similarity_matrix(template_clusters, clusters)
    changed = Vector{Bool}(undef, length(clusters))

    for _ in 1:length(template_clusters)
        max_similarity, max_index = findmax(similarity_matrix)
        if max_similarity == 0.0
            break
        end
        cluster_index, template_index = Tuple(max_index)

        template_id = template_clusters[template_index][1]
        clusters[cluster_index] = (template_id, clusters[cluster_index][2])
        changed[cluster_index] = true

        similarity_matrix[cluster_index, :] .= 0.0
        similarity_matrix[:, template_index] .= 0.0
    end

    max_ID = maximum([cluster[1] for cluster in template_clusters])
    counter = 0
    for i in 1:length(clusters)
        if !changed[i]
            clusters[i] = (max_ID + counter, clusters[i][2])
            counter += 1
        end
    end
end

function best_k_elbow(opinions, max_clusters::Int)
    # Calculate the sum of squared errors (SSE) for different numbers of clusters
    sse = zeros(max_clusters)
    for k in 1:max_clusters
        result = Clustering.kmeans(opinions, k)
        sse[k] = result.totalcost
    end

    # Calculate the elbow point
    best_k = 1
    max_slope = 0
    for k in 2:max_clusters-1
        slope = (sse[k-1] - sse[k+1]) / 2
        if slope > max_slope
            max_slope = slope
            best_k = k
        end
    end

    return best_k
end

function best_k_silhouettes(opinions, max_k::Int)
    best_k = 2
    best_silhouette_avg = -1.0
    distances = Distances.pairwise(Distances.Cityblock(), opinions, dims=2)

    for k in 2:max_k
        kmeans_result = Clustering.kmeans(opinions, k)
        assignments = Clustering.assignments(kmeans_result)
        silhouette_vals = Clustering.silhouettes(assignments, distances)
        silhouette_avg = Statistics.mean(silhouette_vals)

        if silhouette_avg > best_silhouette_avg
            best_silhouette_avg = silhouette_avg
            best_k = k
        end
    end

    return best_k
end