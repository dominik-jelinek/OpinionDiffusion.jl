struct Accumulator
	data::Dict{String, Vector{Any}}
	get_metrics::Function
end

Accumulator(get_metrics::Function) = Accumulator(Dict(), get_metrics)

function add_metrics!(accumulator::Accumulator, model::T) where {T<:Abstract_model}
	metrics = accumulator.get_metrics(model)

	data = accumulator.data
	if length(accumulator.data) == 0
		for (key, value) in metrics
			data[key] = [value]
		end
		data["diffusion_step"] = [0]

		return
	end

	for (key, value) in metrics
		push!(data[key], value)
	end
	push!(data["diffusion_step"], length(data["diffusion_step"]))
end

function accumulated_metrics(accumulator::Accumulator)
	return DataFrame(accumulator.data)
end

function get_metrics(model)
	g = get_social_network(model)
	voters = get_voters(model)
	candidates = get_candidates(model)
	can_count = length(candidates)

	histogram = Graphs.degree_histogram(g)
	keyss = collect(keys(histogram))

	votes = get_votes(voters)

	metrics = Dict(
		"min_degrees" => minimum(keyss),
		"avg_degrees" => Graphs.ne(g) * 2 / Graphs.nv(g),
		"max_degrees" => maximum(keyss),
		"avg_edge_dist" => StatsBase.mean(get_edge_distances(g, voters)),
		"clustering_coeff" => Graphs.global_clustering_coefficient(g),
		#"diameter" => Graphs.diameter(g),

		"avg_vote_length" => StatsBase.mean([length(vote) for vote in votes]),
		"unique_votes" => length(unique(votes)),

		"plurality_scores" => plurality_voting(votes, can_count, true),
		"borda_scores" => borda_voting(votes, can_count, true),
		#"copeland_scores" => copeland_voting(votes, can_count),
		"positions" => get_positions(voters, can_count)
	)

	return metrics
end

# ______________________________________________________________________________
# Dataframe utils
# ______________________________________________________________________________

function agg_stats(df, x_col, y_col)
	gdf = groupby(df, x_col)
	y_col_name = string(y_col)
	functions = [mean, std, minimum, maximum]
	y_col_names = [col_name(y_col_name, func) for func in functions]
	cdf = combine(gdf, y_col .=> [apply$func for func in functions] .=> y_col_names)

	if df[1, y_col] isa Vector{<:Number}
		cdf = squeeze(cdf, x_col)
	end

	return cdf
end

function retrieve_variable(df::DataFrame, config_path)
	if length(config_path) == 1
		return df[!, config_path[1]]
	end

	variable_col = df[!, :experiment_config]

	for var in config_path
		if typeof(var) == Int64
			variable_col = [x[var] for x in variable_col]
		else
			variable_col = [getproperty(x, Symbol(var)) for x in variable_col]
		end
	end

	return variable_col
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

function extract_candidates(df, col)
	values = df[!, col]
	return [[val[candidate] for val in values] for candidate in 1:length(values[1])]
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
	new_data = Dict{String,Vector}()

	for col in names(gdf)
		if col == variable
			new_data[col] = gdf[[1], col]
		else
			new_data[col] = [collect(gdf[!, col])]
		end
	end

	return DataFrame(new_data)
end