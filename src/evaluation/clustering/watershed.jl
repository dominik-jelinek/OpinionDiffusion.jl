Base.@kwdef struct Watershed_clustering_config <: Abstract_clustering_config
	round_digits::Int64
end
name(config::Watershed_clustering_config) = "Watershed clustering"

"""
	clustering(voters::Vector{Abstract_voter}, config::Watershed_clustering_config, projections=nothing)

Returns the labels and clusters of the voters using the watershed clustering algorithm.

# Arguments
- `voters::Vector{Abstract_voter}`: The voters in the election.
- `config::Watershed_clustering_config`: The configuration of the clustering algorithm.
- `projections::Matrix{Float64}`: The projections of the voters. If nothing, the projections are calculated from the voters.

# Returns
- `Vector{Int64}`: The labels of the voters.
- `Vector{Vector{Abstract_voter}}`: The clusters of the voters.
"""
function clustering(voters, clustering_config::Watershed_clustering_config, projections)
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
	watershed(density_map::Matrix{Float64}, step_size::Float64)

Finds all basins and then uses watershed algorithm adaptation to expand them in order to define areas that belong to the clusters.

# Arguments
- `density_map::Matrix{Float64}`: The density map of the voters.
- `step_size::Float64`: The step size of the density map.

# Returns
- `Matrix{Int64}`: The label map of the voters.
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

"""
	inbounds(matrix::Matrix, coord::CartesianIndex)

Returns true if the coordinate is inside the matrix.

# Arguments
- `matrix::Matrix`: The matrix.
- `coord::CartesianIndex`: The coordinate.

# Returns
- `Bool`: True if the coordinate is inside the matrix.
"""
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

"""
	relabel!(label_map, label_1, label_2, label_depths, point_depth)

Relabels all points with label_2 to label_1 if the point is closer to label_1 than to label_2.

# Arguments
- `label_map::Matrix{Int64}`: The label map.
- `label_1::Int64`: The label to relabel to.
- `label_2::Int64`: The label to relabel from.
- `label_depths::Vector{Float64}`: The depths of the labels.
- `point_depth::Float64`: The depth of the point.

# Returns
- `Int64`: The label that was relabeled.
- `Vector{CartesianIndex}`: The indices of the points that were relabeled.
"""
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

"""
	init_labels(queue, shape)

Initializes the label map with the labels of the peaks.

# Arguments
- `queue::Vector{Vector{CartesianIndex}}`: The queue of points.
- `shape::Tuple{Int64, Int64}`: The shape of the label map.

# Returns
- `Matrix{Int64}`: The label map.
"""
function init_labels(queue, shape)
	labels = zeros(Int64, shape)

	for (i, idx) in enumerate(queue)
		labels[idx] = i
	end

	return labels
end


find_local_minima(a) = find_local_maxima(-a)
"""
	find_local_maxima(a)

Finds all local maxima in a vector of real values and returns their indices.

# Arguments
- `a::Vector{Float64}`: The vector of real values.

# Returns
- `Vector{Int64}`: The indices of the local maxima.
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
