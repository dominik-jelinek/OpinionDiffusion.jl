"""
	timestamp_vis(model, sampled_voter_ids, reduce_dim_config, clustering_config, interm_calcs=Dict())

Creates a visualization of the model at the current timestamp.

# Arguments
- `model::Model`: The model to visualize.
- `sampled_voter_ids::Vector{Int64}`: The ids of the voters to sample.
- `reduce_dim_config::Reduce_dim_config`: The config to reduce the dimensionality of the opinions.
- `clustering_config::Clustering_config`: The config to cluster the opinions.
- `interm_calcs::Dict`: A dictionary to store intermediate calculations in.

# Returns
- `visualizations::Vector{Figure}`: The visualizations of the model at the current timestamp.
"""
function timestamp_vis(model, sampled_voter_ids, reduce_dim_config, clustering_config, interm_calcs=Dict())
	visualizations = []
	social_network = get_social_network(model)
	voters = get_voters(model)

	# Sampling for conjested or expensive visualizations
	sampled_voters = voters[sampled_voter_ids]
	sampled_opinions = reduce(hcat, get_opinion(sampled_voters))

	# Dimensionality reduction
	projections = reduce_dims(sampled_opinions, reduce_dim_config)

	if haskey(interm_calcs, "prev_projections")
		unify_projections!(interm_calcs["prev_projections"], projections)
	end

	# Clustering
	labels, clusters = clustering(sampled_voters, clustering_config, projections)
	if haskey(interm_calcs, "prev_clusters")
		unify_clusters!(interm_calcs["prev_clusters"], clusters)
		for (label, indices) in clusters
			labels[collect(indices)] .= label
		end
	end

	# Visualizations

	push!(visualizations, draw_election_summary(get_election_summary(get_votes(sampled_voters))))

	title = name(reduce_dim_config) * "_" * name(clustering_config) * "_" * string(length(sampled_voters))
	push!(visualizations, draw_voter_vis(projections, clusters, title))

	f = Figure()
	draw_voter_KDE!(f[1, 1], projections, name(reduce_dim_config) * "_" * string(length(sampled_voters)))
	push!(visualizations, f)

	f = Figure()
	draw_voter_KDE!(f[1, 1], projections, title)
	cluster_graph = get_cluster_graph(model, clusters, labels, projections)
	draw_cluster_graph!(f[1, 1], cluster_graph)
	push!(visualizations, f)

	cluster_metrics = nothing#cluster_graph_metrics(cluster_graph, social_network, voters, 4)
	#println(modularity(cluster_graph, labels))

	push!(visualizations, draw_degree_distr(Graphs.degree_histogram(social_network)))
	push!(visualizations, draw_edge_distances(get_edge_distances(social_network, voters)))

	# if synchronization from previous visualization is needed we can communicate via interm_calcs
	interm_calcs["prev_projections"] = projections
	interm_calcs["prev_clusters"] = clusters

	return visualizations, interm_calcs
end

function save_pdf(plot_function, filename)
	size_inches = (5, 4)
	size_pt = 72 .* size_inches
	f = Figure(resolution=size_pt, fontsize=12)
	plot_function(f[1, 1])
	save("img/" * filename, f, pt_per_unit=1)
end

function draw_voter_vis(projections, clusters, title, exp_dir=Nothing, counter=[0])
	plot = Plots.plot()
	draw_voter_vis!(plot, projections, clusters, title, exp_dir, counter)

	return plot
end

function draw_voter_vis!(plot, projections, clusters, title, exp_dir=Nothing, counter=[0])
	cluster_colors = Colors.distinguishable_colors(maximum([cluster[1] for cluster in clusters]))

	for (label, indices) in clusters
		Plots.scatter!(plot, Tuple(eachrow(projections[:, collect(indices)])), c=cluster_colors[label], label=length(indices), alpha=0.4)
	end

	if exp_dir != Nothing
		Plots.savefig(plot, "$(exp_dir)/images/$(title)_$(counter[1]).png")
	end

	return plot
end

function draw_voter_KDE!(ax, projections, title="Voter Density Map")
	dens = KernelDensity.kde((projections[1, :], projections[2, :]))
	heatmap(ax, dens, colormap=[:blue, :white, :red, :yellow], axis=(; title=title, xlabel="PC1", ylabel="PC2"))
end

function draw_degree_distr(degree_distribution)
	f = Figure()
	draw_degree_distr!(f[1, 1], degree_distribution)
	return f
end

function draw_degree_distr!(ax, degree_distribution)
	sorted = sort(degree_distribution)
	degrees = collect(keys(sorted))

	counts = collect(values(sorted))
	Axis(ax, xscale=log10, yscale=log10, xlabel="Degree (log10)", ylabel="Num. of vertices (log)", title="Degree distribution")
	println(degree_distribution)
	return Makie.lines!(ax, degrees, counts)
end

function get_edge_distances(social_network, voters)
	distances = Vector{Float64}(undef, Graphs.ne(social_network))
	for (i, edge) in enumerate(Graphs.edges(social_network))
		distances[i] = get_distance(voters[Graphs.src(edge)], voters[Graphs.dst(edge)])
	end

	return distances
end

function get_edge_distances2(social_network, voters)
	distances = Vector{Any}(undef, Graphs.ne(social_network))
	for (i, edge) in enumerate(Graphs.edges(social_network))
		distances[i] = (Graphs.src(edge), Graphs.dst(edge), get_distance(voters[Graphs.src(edge)], voters[Graphs.dst(edge)]))
	end

	return distances
end

function draw_edge_distances(distances)
	plot = Plots.plot()
	draw_edge_distances!(plot, distances)

	return plot
end

function draw_degree_cc!(ax, g)
	degrees = Graphs.degree(g)
	ccs = Graphs.local_clustering_coefficient(g)

	unique_degrees, mean_ccs = aggregate_mean(degrees, ccs)

	Axis(ax, xscale=log10, xlabel="Degree (log10)", ylabel="Clustering coefficient", title="Local Clustering Coefficient")
	Makie.scatter!(ax, degrees, ccs, alpha=0.5)
	Makie.lines!(ax, unique_degrees, mean_ccs, linewidth=4, color=2, colormap=:tab10, colorrange=(1, 10), label="Average clustering coefficient")
	Makie.axislegend()
end

function draw_edge_distances!(plot, distances)
	Plots.histogram!(plot, distances,
		title="Edge distance distribution",
		nbins=100,
		xlims=(0.0, 1.0),
		legend=false,
		ylabel="Num. of vertices",
		xlabel="Distance")
end

function get_election_summary(votes::Vector{Vote}; drop_last::Bool=false)
	can_count = candidate_count(votes[1])
	result = zeros(Float64, can_count, can_count)

	for vote in votes
		position = 1
		for i in 1:length(vote) - 1
			bucket = vote[i]
			for _ in bucket
				frac = 1.0 / length(bucket)
				for c in bucket
					result[c, position] += frac
				end
				position += 1
			end
		end

		if !drop_last || length(vote[end]) == 1
			bucket = vote[end]
			for _ in bucket
				frac = 1.0 / length(bucket)
				for c in bucket
					result[c, position] += frac
				end
				position += 1
			end
		end
	end

	return result ./ length(votes)
end

function draw_election_summary_frequencies(election_summary)
	can_count = size(election_summary, 1)
	f = Figure(figure_padding=2)
	axmain = Axis(f[2, 1], ylabel="Candidate ID", xlabel="Rank", yticks=1:can_count, xticks=(1:can_count))

	hm = Makie.heatmap!(axmain, transpose(election_summary), label="Summary heatmap")#[parties[can.party] for can in candidates])
	#Makie.Colorbar(f[1,2], hm)

	ax2 = Axis(f[2, 2], xlabel="% Votes", ylabel="Candidate ID", yticks=1:can_count, xticks=0.0:0.2:1.0)
	ax2.tellwidth = true
	colsize!(f.layout, 2, Relative(2 / 7))
	rowsize!(f.layout, 1, Relative(1 / 3))

	#rowsize!(f.layout, 2, Aspect(2, 1))

	linkyaxes!(ax2, axmain)
	barplot!(ax2, 1:can_count, vec(sum(election_summary, dims=2)), direction=:x)
	hideydecorations!(ax2)


	ax3, bars = barplot(f[1, 1], 1:can_count, vec(transpose(sum(election_summary, dims=1))), direction=:y, axis=(; ylabel="% Votes", xlabel="Candidate ID", xticks=1:can_count, yticks=0.0:0.2:1.0))
	linkxaxes!(ax3, axmain)

	hidexdecorations!(ax3)
	bar_width = 0.57
	ylims!(ax3, low=0)
	xlims!(ax3, low=bar_width, high=can_count + bar_width - 0.13)
	ylims!(ax2; low=bar_width, high=can_count + bar_width - 0.13)
	xlims!(ax2; low=0)

	#legend = Legend(f[1, 2], [bars, bars], ["1","2"])
	#legend.tellwidth = false
	return f
end

function draw_election_summary(election_summary)
	can_count = size(election_summary, 1)
	f = Figure(figure_padding=2)
	axmain = Axis(f[1, 1], ylabel="Candidate ID", xlabel="Rank", yticks=1:can_count, xticks=(1:can_count))

	hm = Makie.heatmap!(axmain, transpose(election_summary), label="Summary heatmap")#[parties[can.party] for can in candidates])
	Makie.Colorbar(f[1,2], hm)

	return f
end

"""
Loads all logs from one experiment and returns dictionary of visualizations
"""
function gather_vis(exp_dir, sampled_voter_ids, dim_reduction_config, clustering_config,)
	timestamps = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(exp_dir) if split(file, "_")[1] == "model"])
	visualizations = []
	interm_calc = Dict()

	for t in timestamps
		model_log = load_model(exp_dir, t)
		visualization, interm_calc = timestamp_vis(model_log, sampled_voter_ids, dim_reduction_config, clustering_config, interm_calc)
		push!(visualizations, visualization)
		#push!(visualizations, stack_visualizations(model_vis2(model_log, sampled_voter_ids, dim_reduction_config, clustering_config)))
	end

	return visualizations
end

function stack_visualizations(visualizations)
	n = length(visualizations)
	return Plots.plot(visualizations..., size=Plots.default(:size) .* (1, n), layout=(n, 1), bottom_margin=10Plots.mm, left_margin=5Plots.mm, legend=true)
end