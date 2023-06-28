### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 86c32615-cdba-41aa-bfca-e5b90563f7f7
using Pkg

# ╔═╡ 481f2e27-0f88-482a-846a-6a31bf38f3ba
Pkg.activate()

# ╔═╡ 9284976e-d474-11eb-2b94-dbe906a08bd7
using Revise

# ╔═╡ c5aa2487-97aa-48bd-b357-4e806a4c41e9
using Profile

# ╔═╡ c51a1af4-f235-40fa-b828-183e10b2f111
using OpinionDiffusion, PlutoUI

# ╔═╡ 419d618c-1ce4-4702-b2b0-9c3d160c245e
using GraphMakie, CairoMakie, Colors

# ╔═╡ 5d7fc256-c98b-4597-bfc0-51e4bb28c2a4
using DataFrames, Statistics

# ╔═╡ 957eb9d7-12d6-4a22-8338-4e8535b54c71
md"# Opinion diffusion"

# ╔═╡ d2923f02-66d7-47de-9801-d4ad99c1230f
md"## Intro to Pluto notebook"

# ╔═╡ 70e131de-4e31-4deb-9346-641e067269e3
md"Pluto is a reactive notebook that, after changing a cell, automatically recalculates all dependant cells upon it.
- Grey cell = executed cell
- Dark yellow cell = unsaved cell that was changed and is holding an old state
- Red cell = cell that doesn't have all its dependencies or caused an error
- We have put some execution barriers in the form of checkmarks to limit the automatic execution of followed cells as the user might want to change variables safely first before time-consuming. It is still better to have reactive notebooks as the notebook is always in the correct state. 
- As of writing this notebook, it is possible to interrupt execution only on Linux.
- For a faster restart of the notebook, you can press Pluto.jl logo up on top and press black x next to running notebook."

# ╔═╡ 581dab76-bc3e-48f5-8740-6e8cccdcc8d8
md"## Initialize packages"

# ╔═╡ cad5148c-8df4-45ca-a8c0-da29dcddbb8e
TableOfContents(depth=4)

# ╔═╡ 75588cfd-52c9-4406-975c-c03158db6e78
import Distributions, Distances, Graphs, Plots, Random, JLD2

# ╔═╡ 55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
md"Choose visualization backend from Plots library. Plotly for interactivity. GR for speed."

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
Plots.plotly()
#Plots.gr()

# ╔═╡ 746f6fca-1733-4383-b423-f78c3ce80d6a
md"## Config Templates"

# ╔═╡ bdb42e41-d108-41cf-81dc-a5f7cf7b7399
#=
if input_filename == "ED-00001-00000001.toc"
	remove_candidates = [3, 5, 8, 11]
elseif input_filename == "ED-00001-00000002.toc"
	remove_candidates = [8]
elseif input_filename == "ED-00001-00000003.toc"
	remove_candidates = [3, 8, 9, 11]
else
	remove_candidates = []
end
=#

# ╔═╡ 867bb4cb-7edc-4814-ad31-a7cfa403a289
#=
Selection_config(
	remove_candidates = [8],
	rng_seed = rand(UInt32),
	sample_size = 1500
)
=#

# ╔═╡ a02c3b7d-b69a-4f34-91ee-5a735340715d
#=
Spearman_voter_init_config(
	weighting_rate = 0.0
)

SP_diff_init_config(
	rng_seed=rand(UInt32),
	stubbornness_distr=Distributions.Normal(0.5, 0.1)
)

SP_diff_config(
	rng=Random.MersenneTwister(rand(UInt32)),
	evolve_vertices=1.0,
	attract_proba = 1.0,
	change_rate = 0.05,
	normalize_shifts = true
)


Kendall_voter_init_config(
)

KT_diff_init_config(
	rng_seed=rand(UInt32),
	stubbornness_distr=Distributions.Normal(0.5, 0.1)
)

KT_diff_config(
	rng=Random.MersenneTwister(rand(UInt32)),
	evolve_vertices=1.0,
	attract_proba = 1.0
)
=#

# ╔═╡ 5faca0e5-7fef-42fd-a659-226dc74c0ad6
#=
DEG_graph_config(
	rng_seed=rand(UInt32),
	target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
	target_cc=0.3,
	homophily=0.8
)

BA_graph_config(
	rng_seed=rand(UInt32),
	average_degree=10,
	homophily=0.8
)

Random_graph_config(
    rng_seed=rand(UInt32),
    average_degree=10
)

graph_diffusion = Graph_diff_config(
	rng=Random.MersenneTwister(rand(UInt32)),
	evolve_edges=0.0,
	homophily = 0.8
)
=#

# ╔═╡ 51fc614b-5fae-4357-b3fe-537458060109
md"""## Experiments
"""

# ╔═╡ 6e796169-f1bc-4d7f-a23a-4f18a0d5a664
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
        "avg_edge_dist" => mean(get_edge_distances(g, voters)),
        "clustering_coeff" => Graphs.global_clustering_coefficient(g),
        #"diameter" => Graphs.diameter(g),
        
        "avg_vote_length" => mean([length(vote) for vote in votes]),
        "unique_votes" => length(unique(votes)),
        
        "plurality_scores" => plurality_voting(votes, can_count, true),
        "borda_scores" => borda_voting(votes, can_count, true),
        #"copeland_scores" => copeland_voting(votes, can_count),
        "positions" => get_positions(voters, can_count)
	)
	
	return metrics
end

# ╔═╡ b49566b8-e2bf-4e08-b655-ccc29309690a
md"""Select dataset: $(@bind input_filename Select([filename for filename in readdir("./data") if split(filename, ".")[end] == "toc"], default = "ED-00001-00000002.toc"))"""

# ╔═╡ bd34a8b6-193c-4456-af08-c8462bff23ad
if input_filename == "ED-00001-00000001.toc"
	remove_candidates = [3, 5, 8, 11]
elseif input_filename == "ED-00001-00000002.toc"
	remove_candidates = [8]
elseif input_filename == "ED-00001-00000003.toc"
	remove_candidates = [3, 8, 9, 11]
else
	remove_candidates = []
end

# ╔═╡ 85baf553-6584-4940-9c48-ed187e075705
data_dir_path = "./data/"

# ╔═╡ 7e2feb03-1ab4-454e-ad9d-a46e5215d0d6
election = parse_data(data_dir_path * input_filename)

# ╔═╡ eabc778b-5af6-47d5-97ef-c6156786ef19
md"## Model"

# ╔═╡ 978aca85-e644-40f3-ab0c-4dddf1c77b80
md"### Clustering Coefficient"

# ╔═╡ 30887bdf-9bcb-46ad-b93c-63ad0eb078f7
ensemble_sample_graph = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = 42,
			sample_size = x
		) for x in 100:200:2100
	],
	
	voter_init_configs = [
		Spearman_voter_init_config(
			weighting_rate = 0.0
		)
	],
	
	graph_init_configs = vcat(
		[
			DEG_graph_config(
				rng_seed=rand(UInt32),
				target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
				target_cc=0.3,
				homophily=0.0
			) for _ in 1:5
		],
		[
			DEG_graph_config(
				rng_seed=rand(UInt32),
				target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
				target_cc=0.3,
				homophily=0.5
			) for _ in 1:5
		],
		[
			DEG_graph_config(
				rng_seed=rand(UInt32),
				target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
				target_cc=0.3,
				homophily=1.0
			) for _ in 1:5
		],
		[
			BA_graph_config(
				rng_seed=rand(UInt32),
				average_degree=18,
				homophily=0.0
			) for _ in 1:5
		],
		[
			BA_graph_config(
				rng_seed=rand(UInt32),
				average_degree=18,
				homophily=0.5
			) for _ in 1:5
		],
		[
			BA_graph_config(
				rng_seed=rand(UInt32),
				average_degree=18,
				homophily=1.0
			) for _ in 1:5
		],
		[
			Random_graph_config(
		    	rng_seed=rand(UInt32),
		    	average_degree=18
			) for _ in 1:5
		]
	),
	
	diffusions = 0,
	
	diff_init_configs = [],
	
	diff_configs = []
)

# ╔═╡ 3c830004-1902-406c-8984-27bfb0318722
md"""
---
**Execution barrier** $(@bind cb_cc CheckBox())
	
---
"""

# ╔═╡ 5298f733-de5b-4d59-83e3-063eb5a546ba
if cb_cc
	sample_graph_df = ensemble(election, ensemble_sample_graph, get_metrics)
	JLD2.jldsave("logs/ensemble_sample_graph_homophily.jld2"; config=ensemble_sample_graph, df=sample_graph_df)
end

# ╔═╡ b3007e9c-7b02-40e7-8666-e485bbe6ab05
md"#### Graph Type"

# ╔═╡ eea8d995-ca89-43f0-abdf-ee546f643704


# ╔═╡ 9077caca-08cc-4371-91ab-c84e58815bf8
md"#### Homophily"

# ╔═╡ 313f240c-9031-4721-89d8-2c302ef82b89


# ╔═╡ 5745da71-6421-41d9-aa2f-1626a3892ea0
md"## Diffusion"

# ╔═╡ 48f5e7a0-260b-4153-a443-e4962a57861c
md"### Voter Type"

# ╔═╡ 1e6bb760-6a9c-4e6a-85b6-e2ec97653135
ensemble_config_sample_result_SP = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = rand(UInt32),
			sample_size = x
		) for x in 100:200:2100
	],
	
	voter_init_configs = [
		Spearman_voter_init_config(
			weighting_rate = 0.0
		),
		Spearman_voter_init_config(
			weighting_rate = 1.0
		)
	],
	
	graph_init_configs = vcat(
		[
			DEG_graph_config(
				rng_seed=rand(UInt32),
				target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
				target_cc=0.3,
				homophily=0.0
			) for _ in 1:5
		],
		[
			BA_graph_config(
				rng_seed=rand(UInt32),
				average_degree=18,
				homophily=0.0
			) for _ in 1:5
		],
		[
			Random_graph_config(
		    	rng_seed=rand(UInt32),
		    	average_degree=18
			) for _ in 1:5
		]
	),
	
	diffusions = 250,
	
	diff_init_configs = [
		[
			SP_diff_init_config(
				rng_seed=rand(UInt32),
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diff_configs = [
		[
			SP_diff_config(
				rng=Random.MersenneTwister(rand(UInt32)),
				evolve_vertices=1.0,
				attract_proba = 1.0,
				change_rate = 0.05,
				normalize_shifts = true
			)
		]
	]
)

# ╔═╡ 59df032d-e540-4daa-88f3-2cd27b35794c
ensemble_config_sample_result_KT = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = 42,
			sample_size = x
		) for x in 100:200:2100
	],
	
	voter_init_configs = [
		Kendall_voter_init_config(
		),
	],
	
	graph_init_configs = vcat(
		[
			DEG_graph_config(
				rng_seed=rand(UInt32),
				target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
				target_cc=0.3,
				homophily=0.0
			) for _ in 1:5
		],
		[
			BA_graph_config(
				rng_seed=rand(UInt32),
				average_degree=18,
				homophily=0.0
			) for _ in 1:5
		],
		[
			Random_graph_config(
		    	rng_seed=rand(UInt32),
		    	average_degree=18
			) for _ in 1:5
		]
	),
	
	diffusions = 250,
	
	diff_init_configs = [
		[
			KT_diff_init_config(
				rng_seed=rand(UInt32),
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diff_configs = [
		[
			KT_diff_config(
				rng=Random.MersenneTwister(rand(UInt32)),
				evolve_vertices=1.0,
				attract_proba = 1.0
			)
		]
	]
)

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
---
**Execution barrier** $(@bind cb_run CheckBox())
	
---
"""

# ╔═╡ 03bd1c60-4389-4b52-90f4-d1ddf8ef9158
md"Uses DEG graph for the clustering coefficient"

# ╔═╡ 1cce7716-74eb-4766-a678-51bf46d588fd
md"#### Unique Votes"

# ╔═╡ a1cbb7fb-fb0a-46a4-a8f6-0bfed218cb3d
md"#### Average Vote Length"

# ╔═╡ 7d5dc268-1019-4632-becb-4ab8f10ecd61
md"#### Influence Direction"

# ╔═╡ bacc2f88-a732-4c00-98f9-61c3b9eb2531
md"### Diffusion Result"

# ╔═╡ 9d0cc283-820b-43e4-83a7-7f6c8f353b74
md"std for each"

# ╔═╡ b642ebb6-469b-43e3-aa77-b88ab419126f
md"#### Impact of Sample Size"

# ╔═╡ 8940150e-f07c-4f9d-93c8-5e872fb8bc36
md"#### Impact of Graph Type"

# ╔═╡ 8c7c17b3-6b84-4e20-b9c5-84e18227f239
md"#### Impact of Activation Order"

# ╔═╡ 3f1c264b-4aee-427f-8dfd-da5673072e0b
md"#### Impact of Influence Direction"

# ╔═╡ 07689f7d-205b-4000-8673-d45e177815b4
md"#### Impact of Weights in Spearman"

# ╔═╡ 11d3e2b1-5c8a-4aea-b366-4427126d2e65
md"difference of mean, w/o vs with"

# ╔═╡ 859706a0-7d68-4379-ad86-654296d29d7f
md"#### Impact of Stubbornness"

# ╔═╡ 093fc6d3-5d33-4ca3-9a00-c5d8cb0f430b
md"difference of mean, w/o vs with"

# ╔═╡ ac3e648a-1d70-408a-b151-6f5c099f404e
md"
- (Extract), (filter), (groupby), compare
- Extract, filter, draw metric
- Extract, filter, draw voting rule

1. extract from configs used variables or create a column consisting of the type of configs in the column
2. filter a subset of dataframe
3. create dataframe groups to compare

"

# ╔═╡ 4e06848c-4c0a-4441-9894-715f622ccf4e
begin
	extract!(sample_graph_df, "selection_config", "sample_size")
	extract!(sample_graph_df, "graph_init_config", typeof)
end

# ╔═╡ f2cd6477-977e-41af-ac92-acd8aef59197
compare(sample_graph_df, col_name("graph_init_config", typeof), "sample_size",  "clustering_coeff")

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_run
	sample_result_df = ensemble(election, ensemble_config_sample_result, get_metrics)
	JLD2.jldsave("logs/ensemble_sample_result.jld2"; config=ensemble_config_sample_result, df=sample_result_df)
end

# ╔═╡ a83aafb3-7bb3-4dc6-a610-062abdb9b402
#selection = filter(row -> row[:diffusion_step] == maximum(sample_result_df[!, :diffusion_step]), sample_result_df)

# ╔═╡ 7b11bf1f-4386-42a9-9e28-d2720e72c170
#selection = filter(row -> row["sample_size"] == 1500, sample_result_df)

# ╔═╡ a2c1ef1a-285f-4bef-8c70-ae7813699111
selection2 = filter(row -> typeof(row["graph_init_config"]) != Random_graph_config, sample_result_df)

# ╔═╡ 2655e2ff-a481-47cf-a258-cafab8b0e5bb
extract!(selection2, "graph_init_config", "homophily")

# ╔═╡ b96cc7ee-a99d-4c26-af88-6943560ec737
by_homophily = groupby(selection2, "homophily")

# ╔═╡ 1b170978-f757-4969-b0d2-5fee2e008add
function compare2(gdf, col_x, col_y; top_labels=nothing, sub_labels=nothing, linestyles=:solid)
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=col_name(col_x), ylabel=col_name(col_y))
	
	linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
	for (i, sub_group) in enumerate(gdf)
		by_graph_type = groupby(sub_group, col_name("graph_init_config", typeof))
		sub_labels = [split(split(string(key[1]), ".")[2], "_")[1] for (key, _) in pairs(by_graph_type)]
		
		labels = [top_labels[i] * ", " * sub_label for sub_label in sub_labels]
		compare!(ax, by_graph_type, col_x, col_y, labels=labels, linestyle=linestyles[i])
	end

	return f
end

# ╔═╡ 8db73e35-19e5-4381-a99a-492e723da1d8
compare2(by_homophily, "sample_size", "clustering_coeff", top_labels=[string(key[1]) for (key, _) in pairs(by_homophily)])

# ╔═╡ 94f742de-6e68-41ce-84ba-4fe86bec9db7
begin
	extract!(sample_result_df, "selection_config", "sample_size")
	extract!(sample_result_df, "graph_init_config", typeof)
	
end

# ╔═╡ 320389fa-f2ac-4d46-9e21-f54bff4f7274
compare(by_graph_type, "diffusion_step", "unique_votes", labels=labels)

# ╔═╡ a016ed8a-49a4-42b4-96f0-503bfee98cc5
begin
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

# ╔═╡ 2d9df3e4-3532-4d3c-aa7c-407949f7df4a
begin
	extract!(dff, :selection_config, "sample_size")
	voting_rule_vis(agg_stats(dff, "sample_size", "plurality_scores"), "sample_size", "plurality_scores")
end

# ╔═╡ a210fc8f-5d85-464d-8b0b-3fba19579a56
md"## Diffusion with set seed"

# ╔═╡ 94c8e845-2069-4a49-b109-8c71f1ffa2cd
md"Identify interesting runs of diffusion"

# ╔═╡ 3fe9215c-ba2f-4aaa-bcb0-1eb8f1982db8
md"""
---
**Execution barrier** $(@bind anal_run CheckBox())

---
"""

# ╔═╡ ebb18c60-75bf-45dd-8a20-3d402aed23ee
md"Load specific config and save all the logs for in depth analysis."

# ╔═╡ 1173f5f8-b355-468c-8bfb-beebff5ba12b
function extreme_runs(result, metric, can)
	values = [run["metrics"][metric][end][can] for run in result]
	return argmin(values), argmax(values)
end

# ╔═╡ 0cf5fa96-ff32-4853-b6c7-ef1843f5281f
function extreme_runs(result, metric)
	values = [run["metrics"][metric][end] for run in result]
	return argmin(values), argmax(values)
end

# ╔═╡ a4f875d7-685e-43bb-a806-be6ae6547ffb
extremes = extreme_runs(result, "plurality_votings", 4)

# ╔═╡ 3eb2368f-5cdf-4f43-84fa-478865936371
min_model_seed, min_diffusion_seed = result[extremes[1]]["model_seed"], result[extremes[1]]["diffusion_seed"]

# ╔═╡ 5cd2dbf8-5a55-4690-b16d-8b1432015054
max_model_seed, max_diffusion_seed = result[extremes[2]]["model_seed"], result[extremes[2]]["diffusion_seed"]

# ╔═╡ c86a8e9f-c5e9-4931-8923-dcf114df3118
if anal_run
	min_logger = OpinionDiffusion.run(election, candidates, model_config, min_model_seed, diffusion_config, diffusions, min_diffusion_seed)

	max_logger = OpinionDiffusion.run(election, candidates, model_config, max_model_seed, diffusion_config, diffusions, max_diffusion_seed)
end

# ╔═╡ 9883316d-c845-4a4d-a4f8-b8022727cb0b
min_log_idxs = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(min_logger.exp_dir) if split(file, "_")[1] == "model"])

# ╔═╡ 187da043-ee48-420a-91c7-5e3a4fdc30bb
function draw_voting_rules(data, voting_rules, candidates, parties)
	n = length(voting_rules)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)

	for (i, metric) in enumerate(voting_rules)
		result = transpose(reduce(hcat, data[metric]))
		OpinionDiffusion.draw_voting_res!(plot[i, 1], candidates, parties, result, metric)
	end
	
	return Plots.plot(plot, legend=false)
end

# ╔═╡ ff89eead-5bf5-4d52-9134-aa415ae156b7
function compare_voting_rules(logs, voting_rules, candidates, parties)
	if length(logs) == 0
		return
	end
	
	data = [OpinionDiffusion.load(log, "gathered_metrics") for log in logs]
	
	n = length(voting_rules)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)

	linestyles = [:solid, :dot, :dash, :dashdot]
	for (i, metric) in enumerate(voting_rules)
		for (j, dato) in enumerate(data)
			result = transpose(reduce(hcat, dato[metric]))
			OpinionDiffusion.draw_voting_res!(plot[i, 1], candidates, parties, result, metric, linestyle=linestyles[j], log_idx=string(j))
		end
	end
	
	return Plots.plot(plot)
end

# ╔═╡ e70e36ad-066e-4619-a06a-56e325745a0e
function draw_metrics_vis(data, metrics)
	n = length(metrics)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)
	
	for (i, metric) in enumerate(metrics)
		OpinionDiffusion.draw_metric!(plot[i, 1], data[metric], metric)
	end

	return Plots.plot(plot)
end

# ╔═╡ b94c9f37-b321-4db2-9da9-75917be8e52e
function compare_metrics_vis(logs, metrics)
	if length(logs) == 0
		return
	end
	data = [OpinionDiffusion.load(log, "gathered_metrics") for log in logs]
	
	n = length(metrics)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)
	linestyles = [:solid, :dot, :dash, :dashdot]
	for (i, metric) in enumerate(metrics)
		for (j, dato) in enumerate(data)
			OpinionDiffusion.draw_metric!(plot[i, 1], dato[metric], metric, linestyle=linestyles[j], log_idx=j)
		end
	end

	Plots.plot(plot)
end

# ╔═╡ Cell order:
# ╟─957eb9d7-12d6-4a22-8338-4e8535b54c71
# ╟─d2923f02-66d7-47de-9801-d4ad99c1230f
# ╟─70e131de-4e31-4deb-9346-641e067269e3
# ╟─581dab76-bc3e-48f5-8740-6e8cccdcc8d8
# ╠═c5aa2487-97aa-48bd-b357-4e806a4c41e9
# ╠═86c32615-cdba-41aa-bfca-e5b90563f7f7
# ╠═481f2e27-0f88-482a-846a-6a31bf38f3ba
# ╠═9284976e-d474-11eb-2b94-dbe906a08bd7
# ╠═c51a1af4-f235-40fa-b828-183e10b2f111
# ╠═cad5148c-8df4-45ca-a8c0-da29dcddbb8e
# ╠═75588cfd-52c9-4406-975c-c03158db6e78
# ╟─55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
# ╠═ff873978-d93f-4ba2-aadd-6cfd3b136e3d
# ╠═419d618c-1ce4-4702-b2b0-9c3d160c245e
# ╠═5d7fc256-c98b-4597-bfc0-51e4bb28c2a4
# ╟─746f6fca-1733-4383-b423-f78c3ce80d6a
# ╠═bdb42e41-d108-41cf-81dc-a5f7cf7b7399
# ╠═867bb4cb-7edc-4814-ad31-a7cfa403a289
# ╠═a02c3b7d-b69a-4f34-91ee-5a735340715d
# ╠═5faca0e5-7fef-42fd-a659-226dc74c0ad6
# ╟─51fc614b-5fae-4357-b3fe-537458060109
# ╠═6e796169-f1bc-4d7f-a23a-4f18a0d5a664
# ╟─b49566b8-e2bf-4e08-b655-ccc29309690a
# ╠═bd34a8b6-193c-4456-af08-c8462bff23ad
# ╠═85baf553-6584-4940-9c48-ed187e075705
# ╠═7e2feb03-1ab4-454e-ad9d-a46e5215d0d6
# ╟─eabc778b-5af6-47d5-97ef-c6156786ef19
# ╟─978aca85-e644-40f3-ab0c-4dddf1c77b80
# ╟─30887bdf-9bcb-46ad-b93c-63ad0eb078f7
# ╟─3c830004-1902-406c-8984-27bfb0318722
# ╠═5298f733-de5b-4d59-83e3-063eb5a546ba
# ╟─b3007e9c-7b02-40e7-8666-e485bbe6ab05
# ╠═eea8d995-ca89-43f0-abdf-ee546f643704
# ╟─9077caca-08cc-4371-91ab-c84e58815bf8
# ╠═313f240c-9031-4721-89d8-2c302ef82b89
# ╟─5745da71-6421-41d9-aa2f-1626a3892ea0
# ╟─48f5e7a0-260b-4153-a443-e4962a57861c
# ╟─1e6bb760-6a9c-4e6a-85b6-e2ec97653135
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╟─59df032d-e540-4daa-88f3-2cd27b35794c
# ╟─03bd1c60-4389-4b52-90f4-d1ddf8ef9158
# ╟─1cce7716-74eb-4766-a678-51bf46d588fd
# ╟─a1cbb7fb-fb0a-46a4-a8f6-0bfed218cb3d
# ╠═7d5dc268-1019-4632-becb-4ab8f10ecd61
# ╟─bacc2f88-a732-4c00-98f9-61c3b9eb2531
# ╠═9d0cc283-820b-43e4-83a7-7f6c8f353b74
# ╟─b642ebb6-469b-43e3-aa77-b88ab419126f
# ╟─8940150e-f07c-4f9d-93c8-5e872fb8bc36
# ╟─8c7c17b3-6b84-4e20-b9c5-84e18227f239
# ╟─3f1c264b-4aee-427f-8dfd-da5673072e0b
# ╟─07689f7d-205b-4000-8673-d45e177815b4
# ╠═11d3e2b1-5c8a-4aea-b366-4427126d2e65
# ╟─859706a0-7d68-4379-ad86-654296d29d7f
# ╠═093fc6d3-5d33-4ca3-9a00-c5d8cb0f430b
# ╠═ac3e648a-1d70-408a-b151-6f5c099f404e
# ╠═4e06848c-4c0a-4441-9894-715f622ccf4e
# ╠═f2cd6477-977e-41af-ac92-acd8aef59197
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═a83aafb3-7bb3-4dc6-a610-062abdb9b402
# ╠═7b11bf1f-4386-42a9-9e28-d2720e72c170
# ╠═a2c1ef1a-285f-4bef-8c70-ae7813699111
# ╠═2655e2ff-a481-47cf-a258-cafab8b0e5bb
# ╠═b96cc7ee-a99d-4c26-af88-6943560ec737
# ╠═1b170978-f757-4969-b0d2-5fee2e008add
# ╠═8db73e35-19e5-4381-a99a-492e723da1d8
# ╠═94f742de-6e68-41ce-84ba-4fe86bec9db7
# ╠═320389fa-f2ac-4d46-9e21-f54bff4f7274
# ╠═a016ed8a-49a4-42b4-96f0-503bfee98cc5
# ╠═2d9df3e4-3532-4d3c-aa7c-407949f7df4a
# ╟─a210fc8f-5d85-464d-8b0b-3fba19579a56
# ╟─94c8e845-2069-4a49-b109-8c71f1ffa2cd
# ╠═a4f875d7-685e-43bb-a806-be6ae6547ffb
# ╠═3eb2368f-5cdf-4f43-84fa-478865936371
# ╠═5cd2dbf8-5a55-4690-b16d-8b1432015054
# ╟─3fe9215c-ba2f-4aaa-bcb0-1eb8f1982db8
# ╟─ebb18c60-75bf-45dd-8a20-3d402aed23ee
# ╠═c86a8e9f-c5e9-4931-8923-dcf114df3118
# ╠═9883316d-c845-4a4d-a4f8-b8022727cb0b
# ╠═1173f5f8-b355-468c-8bfb-beebff5ba12b
# ╠═0cf5fa96-ff32-4853-b6c7-ef1843f5281f
# ╠═187da043-ee48-420a-91c7-5e3a4fdc30bb
# ╠═ff89eead-5bf5-4d52-9134-aa415ae156b7
# ╠═e70e36ad-066e-4619-a06a-56e325745a0e
# ╠═b94c9f37-b321-4db2-9da9-75917be8e52e
