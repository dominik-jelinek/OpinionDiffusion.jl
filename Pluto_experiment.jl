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
md"Pluto is a reactive notebook that after the change of some variable recalculates all dependant cells upon it automatically. 
- Grey cell = executed cell
- Dark yellow cell = unsaved cell that was changed and is waiting for execution by you
- Red cell = cell that doesn't have all its dependencies or caused error
- I have put some execution barriers in form of checkmark to limit automatic execution of followed cells as the user might want to change variables first before time consuming model generation for example. It is still better to have reactive notebooks as the notebook is always in correct state. 
- At the time of writing this notebook it is possible to interrupt execution only on Linux.
- For faster restart of the notebook you can press Pluto.jl logo up on top and press black x right next to running notebook."

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
        "avg_edge_dist" => OpinionDiffusion.StatsBase.mean(OpinionDiffusion.get_edge_distances(g, voters)),
        "clustering_coeff" => Graphs.global_clustering_coefficient(g),
        #"diameter" => Graphs.diameter(g),
        
        "avg_vote_length" => OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes]),
        "unique_votes" => length(unique(votes)),
        
        "plurality_scores" => plurality_voting(votes, can_count, true),
        "borda_scores" => borda_voting(votes, can_count, true),
        #"copeland_scores" => copeland_voting(votes, can_count),
        "positions" => get_positions(voters, can_count)
	)
	
	return metrics
end

# ╔═╡ b8dd27a7-393d-4b6a-9c63-3b559dc0dc1b
md"### Sample Size and Clustering Coefficient"

# ╔═╡ 30887bdf-9bcb-46ad-b93c-63ad0eb078f7
ensemble_config_DEG = Ensemble_config(
	input_filename = "./data/ED-00001-00000002.toc",
	
	selection_configs = [Selection_config(
			remove_candidates = [8],
			rng_seed = rand(UInt32),
			sample_size = x
		) for x in 100:100:1000
	],
	voter_init_configs = [Spearman_voter_init_config(
		weighting_rate = 0.0
	)],
	graph_init_configs = [DEG_graph_config(
		rng_seed=rand(UInt32),
		target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
	    target_cc=0.3,
	    homophily=0.8
	) for _ in 1:5],
	diffusions = 0,
	diff_init_configs = [],
	diff_configs = []
)

# ╔═╡ d8b50736-f793-48dd-b726-77c9941dfb08
ensemble_config_BA = Ensemble_config(
	input_filename = "./data/ED-00001-00000002.toc",
	
	selection_configs = [Selection_config(
			remove_candidates = [8],
			rng_seed = rand(UInt32),
			sample_size = x
		) for x in 100:100:1000],
	voter_init_configs = [Spearman_voter_init_config(
		weighting_rate = 0.0
	)],
	graph_init_configs = [BA_graph_config(
			rng_seed=rand(UInt32),
			average_degree=10,
			homophily=0.8
		) for _ in 1:5],
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
	deg_df = ensemble(ensemble_config_DEG, get_metrics)
	JLD2.jldsave("logs/experiment_DEG_df.jld2"; deg_df)
end

# ╔═╡ cd60a82a-c672-4ffb-a4b9-066cf2ab2a2a
sample_cc = agg_stats(deg_df, :selection_config, :sample_size, :clustering_coeff)

# ╔═╡ ac20ef3a-ae3a-46e6-9cea-184c70b1e0fe
begin
	f1 = Figure()
	OpinionDiffusion.draw_metric(f1[1,1], sample_cc[!, :sample_size], sample_cc[!, :clustering_coeff_mean], band=(sample_cc[!, :clustering_coeff_minimum], sample_cc[!, :clustering_coeff_maximum]), c=7, label="sample_cc", linestyle=:solid)
	f1
end

# ╔═╡ 793469d3-2862-48b0-a45a-16d2164b6609
ensembl_config_weight_result = Ensemble_config(
	input_filename = "./data/ED-00001-00000002.toc",
	
	selection_configs = [Selection_config(
			remove_candidates = [8],
			rng_seed = rand(UInt32),
			sample_size = 1000
	)],
		
	voter_init_configs = [
		Spearman_voter_init_config(
			weighting_rate = weight
		) for weight in [0.0, 1.0, 2.0]
	],

	graph_init_configs = [DEG_graph_config(
			rng_seed=rand(UInt32),
			target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
	    	target_cc=0.3,
	    	homophily=0.8
		) for _ in 1:5
	],
	
	diffusions = 100,
	
	diff_init_configs = [[SP_diff_init_config(
			rng_seed=rand(UInt32),
			stubbornness_distr=Distributions.Normal(0.5, 0.1)
		)]
	],
	
	diff_configs = [[SP_diff_config(
			rng=Random.MersenneTwister(rand(UInt32)),
			evolve_vertices=1.0,
			attract_proba = 1.0,
			change_rate = 0.05,
			normalize_shifts = true
		)]
	]
)

# ╔═╡ 1e6bb760-6a9c-4e6a-85b6-e2ec97653135
ensemble_config_sample_result = Ensemble_config(
	input_filename = "./data/ED-00001-00000002.toc",
	
	selection_configs = [Selection_config(
			remove_candidates = [8],
			rng_seed = 42,
			sample_size = x
	) for x in 100:200:2100],
	
	voter_init_configs = [
		Spearman_voter_init_config(
			weighting_rate = 0.0
		)
	],
	
	graph_init_configs = [DEG_graph_config(
		rng_seed=rand(UInt32),
		target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
		target_cc=0.3,
		homophily=0.8
	), 
	BA_graph_config(
		rng_seed=rand(UInt32),
		average_degree=18,
		homophily=0.0
	),
	
	Random_graph_config(
	    rng_seed=rand(UInt32),
	    average_degree=18
	)],
	
	diffusions = 100,
	
	diff_init_configs = [[SP_diff_init_config(
		rng_seed=rand(UInt32),
		stubbornness_distr=Distributions.Normal(0.5, 0.0)
	)]],
	
	diff_configs = [[SP_diff_config(
		rng=Random.MersenneTwister(rand(UInt32)),
		evolve_vertices=1.0,
		attract_proba = 1.0,
		change_rate = 0.05,
		normalize_shifts = true
	)]]
)

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
---
**Execution barrier** $(@bind cb_run CheckBox())
	
---
"""

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_run
	sample_result_df = ensemble(ensemble_config_sample_result, get_metrics)
	JLD2.jldsave("logs/sample_result_df.jld2"; sample_result_df, ensemble_config_sample_result)
end

# ╔═╡ a83aafb3-7bb3-4dc6-a610-062abdb9b402
dff = filter(row -> row[:diffusion_step] == maximum(sample_result_df[!, :diffusion_step]), sample_result_df)

# ╔═╡ e2c413fa-bc94-425c-b6c7-6c241d5ef9f2
dff

# ╔═╡ 32041ccb-92bb-41f3-8cb0-a203c45e404d
function extract_candidates(df, col)
	values = df[!, col]
	return [[val[candidate] for val in values] for candidate in 1:length(values[1])]
end

# ╔═╡ 4773255b-9761-4fa6-8c7a-4c235f4909eb
function voting_rule_vis!(ax, df, x_col, voting_rule, linestyle=:solid)
	means = extract_candidates(df, voting_rule * "_mean")
	mins = extract_candidates(df, voting_rule * "_minimum")
	maxs = extract_candidates(df, voting_rule * "_maximum")

	for candidate in eachindex(means)
		OpinionDiffusion.draw_metric!(ax, df[!, x_col], means[candidate], band=(mins[candidate], maxs[candidate]), c=candidate, label=string(candidate), linestyle=linestyle)
	end
end

# ╔═╡ a016ed8a-49a4-42b4-96f0-503bfee98cc5
begin
	f = Figure()
	
	x_col = "sample_size"
	extract!(dff, :selection_config, x_col)
	y_col = "borda_scores"
	
	ax = f[1, 1] = Axis(f; xlabel=x_col, ylabel=y_col)

	config_col = "graph_init_config"
	extract!(dff, config_col, typeof)
	gdf = groupby(dff, config_col * "_typeof")
	for (df, linestyle) in zip(gdf, [:solid, :dashdot, :dot]) 
		voting_rule_vis!(ax, agg_stats(df, x_col, y_col), x_col, y_col, linestyle)
	end
	leg = Legend(f[1, 2], ax)
	f
end

# ╔═╡ d40114bf-734c-4a96-8905-aaccd0c80e22
function voting_rule_vis(df, x_col, voting_rule)
	f = Figure()
	ax = f[1, 1] = Axis(f; xlabel=x_col, ylabel=voting_rule)
	voting_rule_vis!(ax, df, x_col, voting_rule)
	leg = Legend(f[1, 2], ax)
	return f
end

# ╔═╡ 2d9df3e4-3532-4d3c-aa7c-407949f7df4a
begin
	extract!(dff, :selection_config, "sample_size")
	voting_rule_vis(agg_stats(dff, "sample_size", "plurality_scores"), "sample_size", "plurality_scores")
end

# ╔═╡ aad0da61-59d2-4429-97fa-6612872bb863
md"## Diffusion analysis"

# ╔═╡ d716423e-7945-4e0a-a6ab-17e0b94c721e
md"### Compounded metrics"

# ╔═╡ 006bf740-9fc1-41dc-982f-dda7c05ec977
begin
	dict = Dict()
	dict["election_matrix"] = []
	for i in 2:length(gathered_metrics["election_matrix"])
		push!(dict["election_matrix"], abs.(gathered_metrics["election_matrix"][i] - gathered_metrics["election_matrix"][i - 1]))
	end
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

# ╔═╡ 0932e410-4a74-42b3-86c5-ac40f6be3543
md"## Dimensionality reduction and clustering"

# ╔═╡ b203aed5-3231-4e19-a961-089c0a4cf8c6
md"Dimensionality reduction for visualisation of high dimensional opinions"

# ╔═╡ 54855884-22b3-42e0-9120-c5f049043899
out_dim = 2

# ╔═╡ 868986e9-09f5-483e-9b8e-11b5ab6082fd
md"""Dimensionality reduction method: $(@bind dim_reduction_method Select(["PCA", "Tsne", "MDS"], default="PCA"))"""

# ╔═╡ 390be9ec-29a1-4138-952e-fc4eb5eb2ecb
if dim_reduction_method == "PCA"
	dim_reduction_config = PCA_dim_reduction_config(out_dim)
elseif dim_reduction_method == "Tsne"
	dim_reduction_config = Tsne_dim_reduction_config(out_dim = out_dim,
        											reduce_dims = 0,
        											max_iter = 3000,
        											perplexity = 100.0)
else# dim_reduction_method == "MDS"
	dim_reduction_config = MDS_dim_reduction_config(out_dim)
end

# ╔═╡ 52893c8c-d1b5-482a-aae7-b3ec5c590b77
md"Clustering for colouring of voters based on their opinions"

# ╔═╡ e60db281-0bd0-4eb6-94fa-4c30766464fd
md"""Clustering method: $(@bind clustering_method Select(["Kmeans", "GM", "Party", "DBSCAN"], default="Party"))"""

# ╔═╡ ca8a24e4-ce3b-47d9-ad39-8507fa910a9d
if clustering_method == "Party"
	clustering_config = Party_clustering_config(candidates)
elseif clustering_method == "Kmeans"
	clustering_config = Kmeans_clustering_config(length(candidates))
elseif clustering_method == "GM"
	clustering_config = GM_clustering_config(length(candidates))
else# clustering_method == "DBSCAN"
	clustering_config = DBSCAN_clustering_config(0.1, 3)
end

# ╔═╡ 87320bc9-e825-4aa8-84ea-9fd75b7ff4fd
md"## Compare ensemble diffusions"

# ╔═╡ af643514-f20f-436b-966d-05021b566b1e
ensemble_logs = [
	"logs/model_ensemble_2022-07-20_12-50-15.jld2",
	"logs/model_ensemble_2022-07-20_17-03-08.jld2"
]

# ╔═╡ 4add7dc5-0af8-4179-a496-a46767cc85ef
[
	"logs/model_2022-05-29_10-20-40/ensemble_2022-05-29_19-12-38.jld2",
	"logs/model_2022-05-29_22-40-22/ensemble_2022-05-29_22-40-51.jld2",
	"logs/model_2022-05-29_22-41-11/ensemble_2022-05-29_22-43-08.jld2"
]

# ╔═╡ fa876197-2401-4b54-8b56-e4abdf8f8801
 #ensemble_logs = ["logs/" * file for file in readdir("logs") if split(file, "_")[2] == "ensemble"][1:end]

# ╔═╡ 5c505323-4487-4433-8ea7-4357b0fb8166
[OpinionDiffusion.load(log, "diffusion_config") for log in ensemble_logs]

# ╔═╡ a68fa858-2751-451d-a8f3-5c34c95e8b04
 #ensemble_logs = [logger.model_dir * "/" * file for file in readdir(logger.model_dir) if split(file, "_")[1] == "ensemble"][2:end]

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

# ╔═╡ 840c2562-c444-4422-9cf8-e82429163627
draw_voting_rules(gathered_metrics, ["plurality_votings", "borda_votings"], candidates, parties)

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

# ╔═╡ fc90baf0-a51d-4f3f-b036-b3bc381b2dbc
compare_voting_rules(ensemble_logs, ["plurality_votings", "borda_votings"], candidates, parties)

# ╔═╡ e70e36ad-066e-4619-a06a-56e325745a0e
function draw_metrics_vis(data, metrics)
	n = length(metrics)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)
	
	for (i, metric) in enumerate(metrics)
		OpinionDiffusion.draw_metric!(plot[i, 1], data[metric], metric)
	end

	return Plots.plot(plot)
end

# ╔═╡ 09d34d24-0fb0-4cc6-8ab6-c0d55b3346d0
draw_metrics_vis(gathered_metrics, ["unique_votes", "avg_vote_length", "avg_edge_dist", "clustering_coeff"])
#draw_metrics_vis(gathered_metrics, ["clustering_coeff"])

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

# ╔═╡ f96c982d-b5db-47f7-91e0-9c9b3b36332f
compare_metrics_vis(ensemble_logs, ["unique_votes", "avg_vote_length", "avg_edge_dist"])

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
# ╟─b8dd27a7-393d-4b6a-9c63-3b559dc0dc1b
# ╠═30887bdf-9bcb-46ad-b93c-63ad0eb078f7
# ╠═d8b50736-f793-48dd-b726-77c9941dfb08
# ╟─3c830004-1902-406c-8984-27bfb0318722
# ╠═5298f733-de5b-4d59-83e3-063eb5a546ba
# ╠═cd60a82a-c672-4ffb-a4b9-066cf2ab2a2a
# ╠═ac20ef3a-ae3a-46e6-9cea-184c70b1e0fe
# ╠═793469d3-2862-48b0-a45a-16d2164b6609
# ╠═1e6bb760-6a9c-4e6a-85b6-e2ec97653135
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═a83aafb3-7bb3-4dc6-a610-062abdb9b402
# ╠═e2c413fa-bc94-425c-b6c7-6c241d5ef9f2
# ╠═a016ed8a-49a4-42b4-96f0-503bfee98cc5
# ╠═2d9df3e4-3532-4d3c-aa7c-407949f7df4a
# ╠═32041ccb-92bb-41f3-8cb0-a203c45e404d
# ╠═d40114bf-734c-4a96-8905-aaccd0c80e22
# ╠═4773255b-9761-4fa6-8c7a-4c235f4909eb
# ╟─aad0da61-59d2-4429-97fa-6612872bb863
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═09d34d24-0fb0-4cc6-8ab6-c0d55b3346d0
# ╠═840c2562-c444-4422-9cf8-e82429163627
# ╠═006bf740-9fc1-41dc-982f-dda7c05ec977
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
# ╟─0932e410-4a74-42b3-86c5-ac40f6be3543
# ╟─b203aed5-3231-4e19-a961-089c0a4cf8c6
# ╠═54855884-22b3-42e0-9120-c5f049043899
# ╟─868986e9-09f5-483e-9b8e-11b5ab6082fd
# ╟─390be9ec-29a1-4138-952e-fc4eb5eb2ecb
# ╟─52893c8c-d1b5-482a-aae7-b3ec5c590b77
# ╟─e60db281-0bd0-4eb6-94fa-4c30766464fd
# ╟─ca8a24e4-ce3b-47d9-ad39-8507fa910a9d
# ╟─87320bc9-e825-4aa8-84ea-9fd75b7ff4fd
# ╠═af643514-f20f-436b-966d-05021b566b1e
# ╠═4add7dc5-0af8-4179-a496-a46767cc85ef
# ╠═fa876197-2401-4b54-8b56-e4abdf8f8801
# ╠═5c505323-4487-4433-8ea7-4357b0fb8166
# ╠═a68fa858-2751-451d-a8f3-5c34c95e8b04
# ╠═f96c982d-b5db-47f7-91e0-9c9b3b36332f
# ╠═fc90baf0-a51d-4f3f-b036-b3bc381b2dbc
# ╠═187da043-ee48-420a-91c7-5e3a4fdc30bb
# ╠═ff89eead-5bf5-4d52-9134-aa415ae156b7
# ╠═e70e36ad-066e-4619-a06a-56e325745a0e
# ╠═b94c9f37-b321-4db2-9da9-75917be8e52e
