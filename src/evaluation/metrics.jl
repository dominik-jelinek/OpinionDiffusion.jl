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

function draw_metric!(ax, x, y; band::Union{Tuple, Nothing}=nothing, color=Makie.wong_colors()[1], label="", linestyle=:solid)
    line = lines!(ax, x, y, linewidth=3, color = color, linestyle=linestyle, label=label)

    if band !== nothing
        min_y, max_y = band
        band!(ax, x, min_y, max_y, color=(color, 0.3), transparency=true)
    end

    return line
end

function draw_metric(ax, x, y; band::Union{Tuple, Nothing}=nothing, c=1, label="", linestyle=:solid)
    color = Colors.distinguishable_colors(c)[c]
    lines(ax, x, y, linewidth=3, color = color, linestyle=linestyle, label=label)

    if band !== nothing
        min_y, max_y = band
        band!(ax, x, min_y, max_y, color=(color, 0.3), transparency=true)
    end
end

#___________________________________________________________________
# ENSEMBLE
#___________________________________________________________________

function agg_stats(df, x_col, y_col)
	gdf = groupby(df, x_col)
    y_col_name = string(y_col)
    functions = [mean, std, minimum, maximum]
    y_col_names = [col_name(y_col_name, func) for func in functions]

	cdf = combine(gdf, y_col .=> [apply $ func for  func in functions] .=> y_col_names)

	if df[1, y_col] isa Vector{Float64}
		cdf = squeeze(cdf, x_col)
	end
	
	return cdf
end

function extract!(df, config_col::Union{String, Symbol}, variable::Union{String, Symbol})
	df[!, variable] = [getproperty(x, Symbol(variable)) for x in df[!, config_col]]
end

function extract!(df, config_col::Union{String, Symbol}, func::Function)
	df[!,string(config_col) * "_" * string(nameof(func))] = [func(x) for x in df[!, config_col]]
end

function agg_stats(df, config, x_col, y_col)
	df = deepcopy(df)
	df[!, x_col] = [getproperty(x, x_col) for x in df[!, config]]
	return agg_stats(df, x_col, y_col)
end

function apply(func, x)
	if x[1] isa Vector
		return vec(func(hcat(x...), dims=2))
	else
		return [func(x)]
	end
end

function squeeze(df, x_col)
	gdf = groupby(df, x_col)
	return vcat([vectorize(df, x_col) for df in gdf]...)
end

function vectorize(gdf, variable::String)
    new_data = Dict{String, Vector}()
    
    for col in names(gdf)
		if col == variable
			new_data[col] = gdf[[1], col]
		else
        	new_data[col] = [collect(gdf[!, col])]
		end
    end

    return DataFrame(new_data)
end

function col_name(col, func)
    return string(col) * "_" * string(func)
end

function col_name(col)
    split_col = split(string(col), "_")
    # capitalize first letter of each word
    for (i, word) in enumerate(split_col)
        split_col[i] = uppercase(word[1]) * word[2:end]
    end
    return join(split_col, " ")
end

function compare(gdf, col_x, col_y; labels=nothing, linestyle=:solid)
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=col_name(col_x), ylabel=col_name(col_y))
	compare!(ax, gdf, col_x, col_y, labels=labels, linestyle=linestyle)

	return f
end

function compare!(ax, gdf, col_x, col_y; labels=nothing, linestyle=:solid)
    colors = Makie.wong_colors()
    
    for (i, df) in enumerate(gdf)
        stats_df = agg_stats(df, col_x, col_y)
        
        line = draw_metric!(
            ax, 
            stats_df[!, col_x], 
            stats_df[!, col_y * "_mean"], 
            band=(stats_df[!, col_y * "_minimum"], stats_df[!, col_y * "_maximum"]), 
            color=colors[i],
            linestyle=linestyle
        )

        if labels !== nothing
            line[:label] = labels[i]
        end
    end
    axislegend(ax)
end

function voting_rule()
    f = Figure()
	
	x_col = "sample_size"
	extract!(dff, "selection_config", x_col)
	y_col = "borda_scores"
	
	ax = f[1, 1] = Axis(f; xlabel=x_col, ylabel=y_col)

	config_col = "graph_init_config"
	extract!(dff, config_col, typeof)
	gdf = groupby(dff, col_name(config_col, typeof))
	for (df, linestyle) in zip(gdf, [:solid, :dashdot, :dot]) 
		voting_rule_vis!(ax, agg_stats(df, x_col, y_col), x_col, y_col, linestyle)
	end
	leg = Legend(f[1, 2], ax)
	f
end

function voting_rule_vis(df, x_col, voting_rule)
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=x_col, ylabel=voting_rule)
	voting_rule_vis!(ax, df, x_col, voting_rule)
	leg = Legend(f[1, 2], ax)
	return f
end

function voting_rule_vis!(ax, df, x_col, voting_rule, linestyle=:solid)
	means = extract_candidates(df, voting_rule * "_mean")
	mins = extract_candidates(df, voting_rule * "_minimum")
	maxs = extract_candidates(df, voting_rule * "_maximum")
    colors = Makie.wong_colors()
	for candidate in eachindex(means)
		draw_metric!(ax, df[!, x_col], means[candidate], band=(mins[candidate], maxs[candidate]), color=colors[candidate], label=string(candidate), linestyle=linestyle)
	end
end

function extract_candidates(df, col)
	values = df[!, col]
	return [[val[candidate] for val in values] for candidate in 1:length(values[1])]
end