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

# ╔═╡ 867bb4cb-7edc-4814-ad31-a7cfa403a289
#=
Selection_config(
	remove_candidates = remove_candidates,
	rng_seed = rand(UInt32),
	sample_size = 1500
)
=#

# ╔═╡ a02c3b7d-b69a-4f34-91ee-5a735340715d
#=
Spearman_voter_config(
	weighting_rate = 0.0
)

SP_mutation_init_config(
	rng_seed=rand(UInt32),
	stubbornness_distr=Distributions.Normal(0.5, 0.1)
)

SP_mutation_config(
	rng=Random.MersenneTwister(rand(UInt32)),
	evolve_vertices=1.0,
	attract_proba = 1.0,
	change_rate = 0.05,
	normalize_shifts = true
)


Kendall_voter_config(
)

KT_mutation_init_config(
	rng_seed=rand(UInt32),
	stubbornness_distr=Distributions.Normal(0.5, 0.1)
)

KT_mutation_config(
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

graph_diffusion = Graph_mutation_config(
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
        "clustering_coefficient" => Graphs.global_clustering_coefficient(g),
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

# ╔═╡ 7ae9d4d7-925f-4cb0-9b37-c0369f3b32c2
input_filename

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
	
	voter_configs = [
		Spearman_voter_config(
			weighting_rate = 0.0
		)
	],
	
	graph_configs = vcat(
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
	
	diffusion_init_configs = nothing,
	
	diffusion_configs = nothing
)

# ╔═╡ 3c830004-1902-406c-8984-27bfb0318722
md"""
---
**Execution barrier** $(@bind cb_cc CheckBox())
	
---
"""

# ╔═╡ 13802e53-a71b-4a0e-9dbf-0320cb342512
log_dir = "./logs/"

# ╔═╡ 9b4f0f82-6bad-43ad-a227-7da959e7116c
sample_graph_path = log_dir * "ensemble_sample_graph.jld2"

# ╔═╡ 94517137-90f9-4384-9cd2-4e6ebbdf2fa9
if cb_cc
	sample_graph_df = ensemble(election, ensemble_sample_graph, get_metrics)
	save_ensemble(ensemble_sample_graph, sample_graph_df, sample_graph_path)
elseif isfile(sample_graph_path)
	_, sample_graph_df = load_ensemble(sample_graph_path)
end

# ╔═╡ b3007e9c-7b02-40e7-8666-e485bbe6ab05
md"#### Graph Type"

# ╔═╡ 4e06848c-4c0a-4441-9894-715f622ccf4e
begin
	extract!(sample_graph_df, "selection_config", "sample_size")
	extract!(sample_graph_df, "graph_config", typeof)
end

# ╔═╡ f4151588-8ea6-45e7-9c83-b8cd33fb820d
begin
	by_graph_type = groupby(sample_graph_df, col_name("graph_config", typeof))
	labels = [name(key[1]) for (key, _) in pairs(by_graph_type)]
	compare(by_graph_type, "sample_size",  "clustering_coefficient", linestyles=[:solid, :dash, :dot], labels=labels)
end

# ╔═╡ 9077caca-08cc-4371-91ab-c84e58815bf8
md"#### Homophily"

# ╔═╡ 90986999-59f8-4717-864e-20300d0d3918
begin
	selection = filter(row -> typeof(row["graph_config"]) != Random_graph_config, sample_graph_df)
	extract!(selection, "graph_config", "homophily")
	by_homophily = groupby(selection, "homophily")
	labels_hom = ["Homophily = " * string(key[1]) for (key, _) in pairs(by_homophily)]
end

# ╔═╡ e544bb0d-7577-4ad6-a961-fd656fdd254b
begin
	by_homophily_DEG = [filter(row -> typeof(row["graph_config"]) == DEG_graph_config, df) for df in by_homophily]
	compare(by_homophily_DEG, "sample_size",  "clustering_coefficient", linestyles=[:solid, :dash, :dot], labels=labels_hom, title="DEG Graph Clustering Coefficient Based on Homophily and Sample Size")
end

# ╔═╡ 22ddc889-129d-4489-934d-740be4334c5c
begin
	by_homophily_BA = [filter(row -> typeof(row["graph_config"]) == BA_graph_config, df) for df in by_homophily]
	compare(by_homophily_BA, "sample_size",  "clustering_coefficient", linestyles=[:solid, :dash, :dot], labels=labels_hom)
end

# ╔═╡ 5745da71-6421-41d9-aa2f-1626a3892ea0
md"## Diffusion"

# ╔═╡ 48f5e7a0-260b-4153-a443-e4962a57861c
md"### Voter Type"

# ╔═╡ a22ab6ed-3e64-41af-a2fc-b7c6a22a6c42
static_seed = 42

# ╔═╡ 1e6bb760-6a9c-4e6a-85b6-e2ec97653135
ensemble_config_sample_result_SP = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = static_seed,
			sample_size = 1500
		)
	],
	
	voter_configs = [
		Spearman_voter_config(
			weighting_rate = 0.0
		)
	],
	
	graph_configs = [
		DEG_graph_config(
			rng_seed=static_seed,
			target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
			target_cc=0.3,
			homophily=0.0
		)
	],
	
	diffusion_init_configs = [
		[
			SP_mutation_init_config(
				rng_seed=static_seed,
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diffusion_configs = [
		Diffusion_config(
			diffusion_steps=250,
			mutation_configs=[
				SP_mutation_config(
					rng=Random.MersenneTwister(static_seed),
					evolve_vertices=1.0,
					attract_proba = 1.0,
					change_rate = 0.05,
					normalize_shifts = true
				)
			]
		) for _ in 1:10
	]
)

# ╔═╡ a1c06a57-20d2-4f4e-a9e2-343a74ff7841
ensemble_config_sample_result_KT = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = static_seed,
			sample_size = 1500
		)
	],
	
	voter_configs = [
		Kendall_voter_config(
		)
	],
	
	graph_configs = [
		DEG_graph_config(
			rng_seed=static_seed,
			target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
			target_cc=0.3,
			homophily=0.0
		)
	],
	
	diffusion_init_configs = [
		[
			KT_mutation_init_config(
				rng_seed=static_seed,
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diffusion_configs = [
		Diffusion_config(
			diffusion_steps=250,
			mutation_configs=[
				KT_mutation_config(
					rng=Random.MersenneTwister(static_seed),
					evolve_vertices=1.0,
					attract_proba = 1.0
				)
			]
		) for _ in 1:10
	]
)

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
---
**Execution barrier** $(@bind cb_voter CheckBox())
	
---
"""

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_voter
	sample_result_KT = ensemble(election, ensemble_config_sample_result_KT, get_metrics)
	sample_result_SP = ensemble(election, ensemble_config_sample_result_SP, get_metrics)
	sample_result_df = hcat(sample_result_SP, sample_result_KT)
end

# ╔═╡ a016ed8a-49a4-42b4-96f0-503bfee98cc5
function compare_voting_rules(gdf, x_col, y_col)
	f = Figure()
	
	ax = f[1, 1] = Axis(f; xlabel=x_col, ylabel=y_col)
	linestyles = [:solid, :dashdot, :dot]
	for (i, df) in enumerate(gdf) 
		voting_rule_vis!(ax, agg_stats(df, x_col, y_col), x_col, y_col, linestyles[i])
	end
	leg = Legend(f[1, 2], ax)
	f
end

# ╔═╡ 5837e6dd-122e-4d92-9e98-e38da9c96735
md"#### Election Result"

# ╔═╡ c13503c0-f620-44aa-9bac-7aabed099ab2


# ╔═╡ e1a67161-5e4c-4bc9-bb43-2ea60ed01e76
compare_voting_rule([sample_result_SP, sample_result_KT], "diffusion_step", "borda_scores")

# ╔═╡ 30950dda-b87f-47c3-ad51-8da72d8c0b1a


# ╔═╡ 03bd1c60-4389-4b52-90f4-d1ddf8ef9158
md"Uses DEG graph for the clustering coefficient"

# ╔═╡ 1cce7716-74eb-4766-a678-51bf46d588fd
md"#### Unique Votes"

# ╔═╡ a5ca4eff-ccb7-4a48-b762-1d31343526a6
compare([sample_result_SP, sample_result_KT], "diffusion_step",  "unique_votes")

# ╔═╡ a1cbb7fb-fb0a-46a4-a8f6-0bfed218cb3d
md"#### Average Vote Length"

# ╔═╡ 37ff3a7d-1d8d-4e10-8d93-9ebc6b0b366a
compare(sample_result_SP, sample_result_KT, "diffusion_step",  "avg_vote_length")

# ╔═╡ bacc2f88-a732-4c00-98f9-61c3b9eb2531
md"### Diffusion Result"

# ╔═╡ 9d0cc283-820b-43e4-83a7-7f6c8f353b74
md"std for each"

# ╔═╡ b642ebb6-469b-43e3-aa77-b88ab419126f
md"#### Impact of Sample Size"

# ╔═╡ 9acf1a32-26e9-49c3-bb8c-9c281a3840d2
ensemble_config_sample_result = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = rand(UInt32),
			sample_size = x
		) for _ in 1:10 for x in 100:200:2100
	],
	
	voter_configs = [
		Spearman_voter_config(
			weighting_rate = 0.0
		)
	],
	
	graph_configs = [
		DEG_graph_config(
			rng_seed=rand(UInt32),
			target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
			target_cc=0.3,
			homophily=0.0
		)
	],
	
	diffusion_init_configs = [
		[
			SP_mutation_init_config(
				rng_seed=rand(UInt32),
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diffusion_configs = [
		Diffusion_config(
			diffusion_steps=250,
			mutation_configs=[
				SP_mutation_config(
					rng=Random.MersenneTwister(rand(UInt32)),
					evolve_vertices=1.0,
					attract_proba = 1.0,
					change_rate = 0.05,
					normalize_shifts = true
				)
			]
		)
	]
)

# ╔═╡ 8940150e-f07c-4f9d-93c8-5e872fb8bc36
md"#### Impact of Graph Type"

# ╔═╡ 81b233eb-ee65-4d28-b2fa-bc56e537be41
ensemble_config_graphType_result = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = rand(UInt32),
			sample_size = 1500
		)
	],
	
	voter_configs = [
		Spearman_voter_config(
			weighting_rate = 0.0
		)
	],
	
	graph_configs = [
		[
			DEG_graph_config(
				rng_seed=rand(UInt32),
				target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
				target_cc=0.3,
				homophily=0.0
			) for _ in 1:5
		];
		[
			BA_graph_config(
				rng_seed=rand(UInt32),
				average_degree=18,
				homophily=0.0
			) for _ in 1:5
		];
		[
			Random_graph_config(
		    	rng_seed=rand(UInt32),
		    	average_degree=18
			) for _ in 1:5
		]
	],
	
	diffusion_init_configs = [
		[
			SP_mutation_init_config(
				rng_seed=rand(UInt32),
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diffusion_configs = [
		Diffusion_config(
			diffusion_steps=250,
			mutation_configs=[
				SP_mutation_config(
					rng=Random.MersenneTwister(rand(UInt32)),
					evolve_vertices=1.0,
					attract_proba = 1.0,
					change_rate = 0.05,
					normalize_shifts = true
				)
			]
		)
	]
)

# ╔═╡ 8c7c17b3-6b84-4e20-b9c5-84e18227f239
md"#### Impact of Activation Order"

# ╔═╡ 1019ae88-0044-4156-87d9-bd89ca11ab50
ensemble_config_diffusionSeed_result = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = rand(UInt32),
			sample_size = 1500
		)
	],
	
	voter_configs = [
		Spearman_voter_config(
			weighting_rate = 0.0
		)
	],
	
	graph_configs = [
		DEG_graph_config(
			rng_seed=rand(UInt32),
			target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
			target_cc=0.3,
			homophily=0.0
		)
	],
	
	diffusion_init_configs = [
		[
			SP_mutation_init_config(
				rng_seed=rand(UInt32),
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diffusion_configs = [
		Diffusion_config(
			diffusion_steps=250,
			mutation_configs=[
				SP_mutation_config(
					rng=Random.MersenneTwister(rand(UInt32)),
					evolve_vertices=1.0,
					attract_proba = 1.0,
					change_rate = 0.05,
					normalize_shifts = true
				)
			]
		) for _ in 1:10
	]
)

# ╔═╡ 3f1c264b-4aee-427f-8dfd-da5673072e0b
md"#### Impact of Influence Direction"

# ╔═╡ f889196d-d64e-4d63-93e2-569be2e4095e


# ╔═╡ 07689f7d-205b-4000-8673-d45e177815b4
md"#### Impact of Weights in Spearman"

# ╔═╡ 11d3e2b1-5c8a-4aea-b366-4427126d2e65
md"difference of mean, w/o vs with"

# ╔═╡ 5d86d9ee-9b05-486f-b2e6-4969c0653c54
ensemble_config_weighting_result = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = rand(UInt32),
			sample_size = 1500
		)
	],
	
	voter_configs = [
		[
			Spearman_voter_config(
				weighting_rate = 0.0
			) for _ in 1:5
		];
		[
			Spearman_voter_config(
				weighting_rate = 1.0
			) for _ in 1:5
		];
		[
			Spearman_voter_config(
				weighting_rate = 2.0
			) for _ in 1:5
		]
	],
	
	graph_configs = [
		DEG_graph_config(
			rng_seed=rand(UInt32),
			target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
			target_cc=0.3,
			homophily=0.0
		)
	],
	
	diffusion_init_configs = [
		[
			SP_mutation_init_config(
				rng_seed=rand(UInt32),
				stubbornness_distr=Distributions.Normal(0.5, 0.0)
			)
		]
	],
	
	diffusion_configs = [
		Diffusion_config(
			diffusion_steps=250,
			mutation_configs=[
				SP_mutation_config(
					rng=Random.MersenneTwister(rand(UInt32)),
					evolve_vertices=1.0,
					attract_proba = 1.0,
					change_rate = 0.05,
					normalize_shifts = true
				)
			]
		)
	]
)

# ╔═╡ 859706a0-7d68-4379-ad86-654296d29d7f
md"#### Impact of Stubbornness"

# ╔═╡ 093fc6d3-5d33-4ca3-9a00-c5d8cb0f430b
md"difference of mean, w/o vs with"

# ╔═╡ 7015811a-1d7e-4bee-80b8-b11e2c004b5a
ensemble_config_stubbornness_result = Ensemble_config(
	input_filename = input_filename,
	
	selection_configs = [
		Selection_config(
			remove_candidates = remove_candidates,
			rng_seed = rand(UInt32),
			sample_size = 1500
		)
	],
	
	voter_configs = [
		Spearman_voter_config(
			weighting_rate = 0.0
		)
	],
	
	graph_configs = [
		DEG_graph_config(
			rng_seed=rand(UInt32),
			target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
			target_cc=0.3,
			homophily=0.0
		)
	],
	
	diffusion_init_configs = [
		[
			[
				SP_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.1)
				)
			] for _ in 1:10
		];
		[
			[
				SP_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.2)
				)
			] for _ in 1:10
		];
		[
			[
				SP_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.3)
				)
			] for _ in 1:10
		]
	],
	
	diffusion_configs = [
		Diffusion_config(
			diffusion_steps=250,
			mutation_configs=[
				SP_mutation_config(
					rng=Random.MersenneTwister(rand(UInt32)),
					evolve_vertices=1.0,
					attract_proba = 1.0,
					change_rate = 0.05,
					normalize_shifts = true
				)
			]
		)
	]
)

# ╔═╡ ac3e648a-1d70-408a-b151-6f5c099f404e
md"
- (Extract), (filter), (groupby), compare
- Extract, filter, draw metric
- Extract, filter, draw voting rule

1. extract from configs used variables or create a column consisting of the type of configs in the column
2. filter a subset of dataframe
3. create dataframe groups to compare

"

# ╔═╡ 94f742de-6e68-41ce-84ba-4fe86bec9db7
begin
	extract!(sample_result_df, "selection_config", "sample_size")
	extract!(sample_result_df, "graph_init_config", typeof)
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
# ╠═867bb4cb-7edc-4814-ad31-a7cfa403a289
# ╠═a02c3b7d-b69a-4f34-91ee-5a735340715d
# ╠═5faca0e5-7fef-42fd-a659-226dc74c0ad6
# ╟─51fc614b-5fae-4357-b3fe-537458060109
# ╠═6e796169-f1bc-4d7f-a23a-4f18a0d5a664
# ╟─b49566b8-e2bf-4e08-b655-ccc29309690a
# ╠═7ae9d4d7-925f-4cb0-9b37-c0369f3b32c2
# ╠═bd34a8b6-193c-4456-af08-c8462bff23ad
# ╠═85baf553-6584-4940-9c48-ed187e075705
# ╠═7e2feb03-1ab4-454e-ad9d-a46e5215d0d6
# ╟─eabc778b-5af6-47d5-97ef-c6156786ef19
# ╟─978aca85-e644-40f3-ab0c-4dddf1c77b80
# ╠═30887bdf-9bcb-46ad-b93c-63ad0eb078f7
# ╠═3c830004-1902-406c-8984-27bfb0318722
# ╠═13802e53-a71b-4a0e-9dbf-0320cb342512
# ╠═9b4f0f82-6bad-43ad-a227-7da959e7116c
# ╠═94517137-90f9-4384-9cd2-4e6ebbdf2fa9
# ╟─b3007e9c-7b02-40e7-8666-e485bbe6ab05
# ╠═4e06848c-4c0a-4441-9894-715f622ccf4e
# ╠═f4151588-8ea6-45e7-9c83-b8cd33fb820d
# ╟─9077caca-08cc-4371-91ab-c84e58815bf8
# ╟─90986999-59f8-4717-864e-20300d0d3918
# ╠═e544bb0d-7577-4ad6-a961-fd656fdd254b
# ╠═22ddc889-129d-4489-934d-740be4334c5c
# ╟─5745da71-6421-41d9-aa2f-1626a3892ea0
# ╟─48f5e7a0-260b-4153-a443-e4962a57861c
# ╠═a22ab6ed-3e64-41af-a2fc-b7c6a22a6c42
# ╠═1e6bb760-6a9c-4e6a-85b6-e2ec97653135
# ╠═a1c06a57-20d2-4f4e-a9e2-343a74ff7841
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═a016ed8a-49a4-42b4-96f0-503bfee98cc5
# ╟─5837e6dd-122e-4d92-9e98-e38da9c96735
# ╠═c13503c0-f620-44aa-9bac-7aabed099ab2
# ╠═e1a67161-5e4c-4bc9-bb43-2ea60ed01e76
# ╠═30950dda-b87f-47c3-ad51-8da72d8c0b1a
# ╟─03bd1c60-4389-4b52-90f4-d1ddf8ef9158
# ╟─1cce7716-74eb-4766-a678-51bf46d588fd
# ╠═a5ca4eff-ccb7-4a48-b762-1d31343526a6
# ╟─a1cbb7fb-fb0a-46a4-a8f6-0bfed218cb3d
# ╠═37ff3a7d-1d8d-4e10-8d93-9ebc6b0b366a
# ╟─bacc2f88-a732-4c00-98f9-61c3b9eb2531
# ╠═9d0cc283-820b-43e4-83a7-7f6c8f353b74
# ╟─b642ebb6-469b-43e3-aa77-b88ab419126f
# ╠═9acf1a32-26e9-49c3-bb8c-9c281a3840d2
# ╟─8940150e-f07c-4f9d-93c8-5e872fb8bc36
# ╠═81b233eb-ee65-4d28-b2fa-bc56e537be41
# ╟─8c7c17b3-6b84-4e20-b9c5-84e18227f239
# ╠═1019ae88-0044-4156-87d9-bd89ca11ab50
# ╟─3f1c264b-4aee-427f-8dfd-da5673072e0b
# ╠═f889196d-d64e-4d63-93e2-569be2e4095e
# ╟─07689f7d-205b-4000-8673-d45e177815b4
# ╟─11d3e2b1-5c8a-4aea-b366-4427126d2e65
# ╠═5d86d9ee-9b05-486f-b2e6-4969c0653c54
# ╟─859706a0-7d68-4379-ad86-654296d29d7f
# ╟─093fc6d3-5d33-4ca3-9a00-c5d8cb0f430b
# ╠═7015811a-1d7e-4bee-80b8-b11e2c004b5a
# ╠═ac3e648a-1d70-408a-b151-6f5c099f404e
# ╠═94f742de-6e68-41ce-84ba-4fe86bec9db7
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
