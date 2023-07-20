"""
	metrics_vis(metrics, candidates, parties, exp_dir=Nothing)

Visualizes the metrics.

# Arguments
- `metrics`: The metrics to visualize.
- `candidates`: The candidates in the election.
- `parties`: The parties in the election.
- `exp_dir=Nothing`: The directory of the experiment.

# Returns
- `Plots.Plot`: The plot of the metrics.
"""
function metrics_vis(metrics, candidates, parties, exp_dir=Nothing)
	degrees = draw_range(metrics["min_degrees"], metrics["avg_degrees"], metrics["max_degrees"], title="Degree range", xlabel="Timestamp", ylabel="Degree", value_label="avg")

	plurality = draw_voting_res(candidates, parties, reduce(hcat, metrics["plurality_votings"])', "Plurality voting")
	borda = draw_voting_res(candidates, parties, reduce(hcat, metrics["borda_votings"])', "Borda voting")
	copeland = draw_voting_res(candidates, parties, reduce(hcat, metrics["copeland_votings"])', "Copeland voting")

	plots = Plots.plot(degrees, plurality, borda, copeland, layout=(2, 2), size=(669, 900))

	if exp_dir != Nothing
		Plots.savefig(plots, "$(exp_dir)/images/metrics.png")
	end

	return plots
end

"""
	gather_metrics(experiment_dir::String, get_metrics::Function)

Gathers the metrics from the models in the experiment directory.

# Arguments
- `experiment_dir::String`: The directory of the experiment.
- `get_metrics::Function`: The function to use to get the metrics from the model.

# Returns
- `Dict{String, Any}`: The metrics gathered from the models.
"""
function gather_metrics(experiment_dir::String, get_metrics::Function)
	timestamps = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(experiment_dir) if split(file, "_")[1] == "model"])

	accumulator = Accumulator(get_metrics)
	for timestamp in timestamps
		add_metrics!(accumulator, load_model(experiment_dir, timestamp))
	end

	return accumulated_metrics(accumulator)
end

"""
	compare_voting_rule(gdf, x_col, y_col; candidates=nothing, linestyles=[:solid], labels=nothing, colors=to_colormap(:tab10))

Compares the voting rules in the given data frame.

# Arguments
- `gdf`: The data frame containing the data.
- `x_col`: The column to use as the x-axis.
- `y_col`: The column to use as the y-axis.
- `candidates=nothing`: The candidates to use in the legend.
- `linestyles=[:solid]`: The line styles to use for the different voting rules.
- `labels=nothing`: The labels to use in the legend.
- `colors=to_colormap(:tab10)`: The colors to use for the different voting rules.
"""
function compare_voting_rule(gdf, x_col, y_col; candidates=nothing, linestyles=[:solid], labels=nothing, colors=to_colormap(:tab10))
	f = Figure()

	ax = f[1, 1] = Axis(f; xlabel=col_name(x_col), ylabel=col_name(y_col))
	compare_voting_rule!(ax, gdf, x_col, y_col, candidates=candidates, linestyles=linestyles, colors=colors)

	if labels !== nothing
		legend_elements = []
		for i in 1:length(gdf)
			linestyle = linestyles[(i-1) % length(linestyles) + 1]
			push!(legend_elements, [LineElement(color = :black, linestyle = linestyle)])
		end
		axislegend(ax, legend_elements, labels, position = :lt)
	end

	if candidates !== nothing
		Legend(f[1, 2], ax)
	end
	f
end

"""
	compare_voting_rule!(ax, gdf, x_col, y_col; candidates=nothing, linestyles=[:solid], colors=to_colormap(:tab20))

Compares the voting rules in the given data frame.

# Arguments
- `ax`: The axis to draw the plot on.
- `gdf`: The data frame containing the data.
- `x_col`: The column to use as the x-axis.
- `y_col`: The column to use as the y-axis.
- `candidates=nothing`: The candidates to use in the legend.
- `linestyles=[:solid]`: The line styles to use for the different voting rules.
- `colors=to_colormap(:tab20)`: The colors to use for the different voting rules.
"""
function compare_voting_rule!(ax, gdf, x_col, y_col; candidates=nothing, linestyles=[:solid], colors=to_colormap(:tab20))
	for (i, df) in enumerate(gdf)
		stats_df = agg_stats(df, x_col, y_col)
		draw_voting_rule!(ax, stats_df, x_col, y_col, candidates=candidates, linestyle=linestyles[(i-1) % length(linestyles) + 1], colors=colors)
	end
end

"""
	draw_voting_rule(stats_df, x_col, y_col; candidates=nothing, linestyle=:solid, colors=to_colormap(:tab20))

Draws the voting rule in the given data frame.

# Arguments
- `stats_df`: The data frame containing the data.
- `x_col`: The column to use as the x-axis.
- `y_col`: The column to use as the y-axis.
- `candidates=nothing`: The candidates to use in the legend.
- `linestyle=:solid`: The line style to use.
- `colors=to_colormap(:tab20)`: The colors to use for the different voting rules.
"""
function draw_voting_rule(stats_df, x_col, y_col; candidates=nothing, linestyle=:solid, colors=to_colormap(:tab20))
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=col_name(x_col), ylabel=col_name(y_col))
	draw_voting_rule!(ax, stats_df, x_col, y_col, candidates=candidates, linestyle=linestyle, colors=colors)
	if candidates !== nothing
		Legend(f[1, 2], ax)
	end
	return f
end

"""
	draw_voting_rule!(ax, stats_df, x_col, y_col; candidates=nothing, linestyle=:solid, colors=to_colormap(:tab20))

Draws the voting rule in the given data frame.

# Arguments
- `ax`: The axis to draw the plot on.
- `stats_df`: The data frame containing the data.
- `x_col`: The column to use as the x-axis.
- `y_col`: The column to use as the y-axis.
- `candidates=nothing`: The candidates to use in the legend.
- `linestyle=:solid`: The line style to use.
- `colors=to_colormap(:tab20)`: The colors to use for the different voting rules.
"""
function draw_voting_rule!(ax, stats_df, x_col, y_col; candidates, linestyle=:solid, colors=to_colormap(:tab20))
	means = extract_candidates(stats_df, y_col * "_mean")
	stds = extract_candidates(stats_df, y_col * "_std")
	mins = means - stds
	maxs = means + stds

	#mins = extract_candidates(stats_df, y_col * "_minimum")
	#maxs = extract_candidates(stats_df, y_col * "_maximum")
	#colors = Makie.wong_colors()
	#colors = Colors.distinguishable_colors(length(means))

	for i in eachindex(means)
		label = candidates === nothing ? string(i) : string(get_ID(candidates[i])) * ". " * get_name(candidates[i]) * ", " * get_party_name(candidates[i])
		draw_metric!(ax, stats_df[!, x_col], means[i], band=(mins[i], maxs[i]), color=colors[i], label=label, linestyle=linestyle)
	end
end

"""
	compare_metric(gdf, x_col, y_col; labels=nothing, linestyles=[:solid], title="")

Compares the metrics in the given data frame.

# Arguments
- `gdf`: The data frame containing the data.
- `x_col`: The column to use as the x-axis.
- `y_col`: The column to use as the y-axis.
- `labels=nothing`: The labels to use in the legend.
- `linestyles=[:solid]`: The line styles to use for the different metrics.
- `title=""`: The title of the plot.
"""
function compare_metric(gdf, x_col, y_col; labels=nothing, linestyles=[:solid], title="")
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=col_name(x_col), ylabel=col_name(y_col))

	compare_metric!(ax, gdf, x_col, y_col, labels=labels, linestyles=linestyles)

	if title != ""
		ax.title = title
	end

	return f
end

"""
	compare_metric!(ax, gdf, x_col, y_col; labels=nothing, linestyles=[:solid])

Compares the metrics in the given data frame.

# Arguments
- `ax`: The axis to draw the plot on.
- `gdf`: The data frame containing the data.
- `x_col`: The column to use as the x-axis.
- `y_col`: The column to use as the y-axis.
- `labels=nothing`: The labels to use in the legend.
- `linestyles=[:solid]`: The line styles to use for the different metrics.
"""
function compare_metric!(ax, gdf, x_col, y_col; labels=nothing, linestyles=[:solid])
	colors = Makie.wong_colors()

	for (i, df) in enumerate(gdf)
		stats_df = agg_stats(df, x_col, y_col)

		line = draw_metric!(
			ax,
			stats_df[!, x_col],
			stats_df[!, y_col * "_mean"],
			band=(stats_df[!, y_col * "_minimum"], stats_df[!, y_col * "_maximum"]),
			color=colors[i],
			linestyle=linestyles[(i-1) % length(linestyles) + 1]
		)

		if labels !== nothing
			line[:label] = labels[i]
		end
	end
	if labels !== nothing
		axislegend(ax)
	end
end

"""
	draw_metric(stats_df, x_col, y; band::Union{Tuple,Nothing}=nothing, color=Makie.wong_colors()[1], label="", linestyle=:solid)

Draws the metric in the given data frame.

# Arguments
- `stats_df`: The data frame containing the data.
- `x_col`: The column to use as the x-axis.
- `y`: The y-values to use.
- `band::Union{Tuple,Nothing}=nothing`: The band to draw.
- `color=Makie.wong_colors()[1]`: The color to use.
- `label=""`: The label to use.
- `linestyle=:solid`: The line style to use.
"""
function draw_metric(stats_df, x_col, y; band::Union{Tuple,Nothing}=nothing, color=Makie.wong_colors()[1], label="", linestyle=:solid)
	line = lines!(ax, x, y, linewidth=3, color=color, linestyle=linestyle, label=label)

	if band !== nothing
		min_y, max_y = band
		band!(ax, x, min_y, max_y, color=(color, 0.3), transparency=true)
	end

	return line
end

"""
	draw_metric!(ax, x, y; band::Union{Tuple,Nothing}=nothing, color=Makie.wong_colors()[1], label="", linestyle=:solid)

Draws the metric in the given data frame.

# Arguments
- `ax`: The axis to draw the plot on.
- `x`: The x-values to use.
- `y`: The y-values to use.
- `band::Union{Tuple,Nothing}=nothing`: The band to draw.
- `color=Makie.wong_colors()[1]`: The color to use.
- `label=""`: The label to use.
- `linestyle=:solid`: The line style to use.
"""
function draw_metric!(ax, x, y; band::Union{Tuple,Nothing}=nothing, color=Makie.wong_colors()[1], label="", linestyle=:solid)
	line = lines!(ax, x, y, linewidth=3, color=color, linestyle=linestyle, label=label)

	if band !== nothing
		min_y, max_y = band
		band!(ax, x, min_y, max_y, color=(color, 0.3), transparency=true)
	end

	return line
end
