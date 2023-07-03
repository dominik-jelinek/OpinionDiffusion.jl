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

function compare(gdf, x_col, y_col; labels=nothing, linestyles=[:solid], title="")
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=col_name(x_col), ylabel=col_name(y_col))

	compare!(ax, gdf, x_col, y_col, labels=labels, linestyles=linestyles)

	if title != ""
		ax.title = title
	end

	return f
end

function compare!(ax, gdf, x_col, y_col; labels=nothing, linestyles=[:solid])
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
	else
		axislegend(ax, position = :lt)
	end

	f
end

function compare_voting_rule!(ax, gdf, x_col, y_col; candidates=nothing, linestyles=[:solid], colors=to_colormap(:tab20))
	for (i, df) in enumerate(gdf)
		stats_df = agg_stats(df, x_col, y_col)
		voting_rule_vis!(ax, stats_df, x_col, y_col, candidates=candidates, linestyle=linestyles[(i-1) % length(linestyles) + 1], colors=colors)
	end
end

function voting_rule_vis(stats_df, x_col, y_col; candidates=nothing, linestyle=:solid, colors=to_colormap(:tab20))
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=x_col, ylabel=y_col)
	voting_rule_vis!(ax, stats_df, x_col, y_col, candidates=candidates, linestyle=linestyle, colors=colors)
	Legend(f[1, 2], ax)
	return f
end

function voting_rule_vis!(ax, stats_df, x_col, y_col; candidates, linestyle=:solid, colors=to_colormap(:tab20))
	means = extract_candidates(stats_df, y_col * "_mean")
	mins = extract_candidates(stats_df, y_col * "_minimum")
	maxs = extract_candidates(stats_df, y_col * "_maximum")
	#colors = Makie.wong_colors()
	#colors = Colors.distinguishable_colors(length(means))

	for i in eachindex(means)
		label = candidates === nothing ? string(i) : string(get_ID(candidates[i])) * ", party:" * string(get_party_ID(candidates[i]))
		draw_metric!(ax, stats_df[!, x_col], means[i], band=(mins[i], maxs[i]), color=colors[i], label=label, linestyle=linestyle)
	end
end

function draw_metric!(ax, x, y; band::Union{Tuple,Nothing}=nothing, color=Makie.wong_colors()[1], label="", linestyle=:solid)
	line = lines!(ax, x, y, linewidth=3, color=color, linestyle=linestyle, label=label)

	if band !== nothing
		min_y, max_y = band
		band!(ax, x, min_y, max_y, color=(color, 0.3), transparency=true)
	end

	return line
end