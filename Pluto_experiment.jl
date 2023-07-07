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
data_path = data_path,
remove_candidate_ids= remove_candidate_ids,

Sampling_config(
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
	remove_candidate_ids = [3, 5, 8, 11]
elseif input_filename == "ED-00001-00000002.toc"
	remove_candidate_ids = [8]
elseif input_filename == "ED-00001-00000003.toc"
	remove_candidate_ids = [3, 8, 9, 11]
else
	remove_candidate_ids = []
end

# ╔═╡ 85baf553-6584-4940-9c48-ed187e075705
data_path = "./data/" * input_filename

# ╔═╡ 5a4937a0-abb0-440b-b9cc-ea43522f6674
election = remove_candidates(parse_data(data_path), remove_candidate_ids)

# ╔═╡ 489639d0-31b0-4da7-8e16-e5d7a33fbd85
linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

# ╔═╡ 13802e53-a71b-4a0e-9dbf-0320cb342512
log_dir = "./logs/"

# ╔═╡ eabc778b-5af6-47d5-97ef-c6156786ef19
md"## Model"

# ╔═╡ 978aca85-e644-40f3-ab0c-4dddf1c77b80
md"### Clustering Coefficient"

# ╔═╡ 30887bdf-9bcb-46ad-b93c-63ad0eb078f7
ensemble_sample_graph = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
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
	
	diffusion_run_configs = nothing
)

# ╔═╡ b3007e9c-7b02-40e7-8666-e485bbe6ab05
md"#### Graph Type"

# ╔═╡ 9077caca-08cc-4371-91ab-c84e58815bf8
md"#### Homophily"

# ╔═╡ 5745da71-6421-41d9-aa2f-1626a3892ea0
md"## Diffusion"

# ╔═╡ 48f5e7a0-260b-4153-a443-e4962a57861c
md"### Voter Type"

# ╔═╡ a22ab6ed-3e64-41af-a2fc-b7c6a22a6c42
static_seed = 42

# ╔═╡ 1e6bb760-6a9c-4e6a-85b6-e2ec97653135
ensemble_config_SP_result = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
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
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=400,
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

# ╔═╡ a1c06a57-20d2-4f4e-a9e2-343a74ff7841
ensemble_config_KT_result = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
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
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=400,
			mutation_configs=[
				KT_mutation_config(
					rng=Random.MersenneTwister(rand(UInt32)),
					evolve_vertices=1.0,
					attract_proba = 1.0
				)
			]
		) for _ in 1:10
	]
)

# ╔═╡ 604d68b0-dff1-45f7-8a5b-ac2eb44949b2
function run(ensemble_config, get_metrics, path; recalculate=false)
	if !recalculate && isfile(path)
		_, dataframe = load_ensemble(path)
		return dataframe
	end
	
	dataframe = ensemble(ensemble_config, get_metrics)
	save_ensemble(ensemble_config, dataframe, path)

	return dataframe
end

# ╔═╡ f8492aca-23ea-406f-87a5-0722f4ee2330
sample_graph_df = run(ensemble_sample_graph, get_metrics, log_dir * "ensemble_sample_graph.jld2", recalculate=false)

# ╔═╡ 4e06848c-4c0a-4441-9894-715f622ccf4e
begin
	sample_graph_df[!, "sample_size"] = retrieve_variable(sample_graph_df, ["election_config", "sampling_config", "sample_size"])

	sample_graph_df[!, "graph_config"] = retrieve_variable(sample_graph_df, ["model_config", "graph_config"])
	
	sample_graph_df[!, "graph_config_typeof"] = typeof.(sample_graph_df[!, "graph_config"])
end

# ╔═╡ f4151588-8ea6-45e7-9c83-b8cd33fb820d
begin
	by_graph_type = groupby(sample_graph_df, "graph_config_typeof")
	labels = [name(key[1]) for (key, _) in pairs(by_graph_type)]
	compare_metric(by_graph_type, "sample_size",  "clustering_coefficient", linestyles=linestyles, labels=labels)
end

# ╔═╡ 90986999-59f8-4717-864e-20300d0d3918
begin
	selection = filter(row -> typeof(row["graph_config"]) != Random_graph_config, sample_graph_df)
	selection[!, "homophily"] = retrieve_variable(selection, ["graph_config", "homophily"])
	by_homophily = groupby(selection, "homophily")
	labels_hom = ["Homophily = " * string(key[1]) for (key, _) in pairs(by_homophily)]
end

# ╔═╡ e544bb0d-7577-4ad6-a961-fd656fdd254b
begin
	by_homophily_DEG = [filter(row -> typeof(row["graph_config"]) == DEG_graph_config, df) for df in by_homophily]
	compare_metric(by_homophily_DEG, "sample_size",  "clustering_coefficient", linestyles=linestyles, labels=labels_hom, title="DEG Graph Clustering Coefficient Based on Homophily and Sample Size")
end

# ╔═╡ 22ddc889-129d-4489-934d-740be4334c5c
begin
	by_homophily_BA = [filter(row -> typeof(row["graph_config"]) == BA_graph_config, df) for df in by_homophily]
	compare_metric(by_homophily_BA, "sample_size",  "clustering_coefficient", linestyles=linestyles, labels=labels_hom)
end

# ╔═╡ 5a7df6ee-e403-48ec-8349-0a29767d1af8
SP_result = run(ensemble_config_SP_result, get_metrics, log_dir * "ensemble_SP_result.jld2", recalculate=false)

# ╔═╡ 9615b9f8-be60-4730-96e7-d6d45e68c346
KT_result = run(ensemble_config_KT_result, get_metrics, log_dir * "ensemble_KT_result.jld2", recalculate=false)

# ╔═╡ 5837e6dd-122e-4d92-9e98-e38da9c96735
md"#### Election Result"

# ╔═╡ 03bd1c60-4389-4b52-90f4-d1ddf8ef9158
md"Uses DEG graph for the clustering coefficient"

# ╔═╡ 6c98c5a0-b3c8-48dc-b848-caa7c6688493
begin
	KT_result[!, "voter_config"] = retrieve_variable(KT_result, ["model_config", "voter_config"])
	SP_result[!, "voter_config"] = retrieve_variable(SP_result, ["model_config", "voter_config"])
end

# ╔═╡ 1cce7716-74eb-4766-a678-51bf46d588fd
md"#### Unique Votes"

# ╔═╡ a5ca4eff-ccb7-4a48-b762-1d31343526a6
compare_metric([SP_result, KT_result], "diffusion_step",  "unique_votes", labels=[name(SP_result.voter_config[1]), name(KT_result.voter_config[1])])

# ╔═╡ a1cbb7fb-fb0a-46a4-a8f6-0bfed218cb3d
md"#### Average Vote Length"

# ╔═╡ 37ff3a7d-1d8d-4e10-8d93-9ebc6b0b366a
compare([SP_result, KT_result], "diffusion_step",  "avg_vote_length", labels=[name(SP_result.voter_config[1]), name(KT_result.voter_config[1])])

# ╔═╡ e1a67161-5e4c-4bc9-bb43-2ea60ed01e76
compare_voting_rule([SP_result, KT_result], "diffusion_step", "borda_scores", linestyles=linestyles, labels=[name(SP_result.voter_config[1]), name(KT_result.voter_config[1])], candidates=get_candidates(election))

# ╔═╡ 30950dda-b87f-47c3-ad51-8da72d8c0b1a
compare_voting_rule([SP_result, KT_result], "diffusion_step", "plurality_scores", linestyles=linestyles, labels=[name(SP_result.voter_config[1]), name(KT_result.voter_config[1])])

# ╔═╡ bacc2f88-a732-4c00-98f9-61c3b9eb2531
md"### Diffusion Result"

# ╔═╡ b642ebb6-469b-43e3-aa77-b88ab419126f
md"#### Impact of Sample Size"

# ╔═╡ 9acf1a32-26e9-49c3-bb8c-9c281a3840d2
ensemble_config_sample_result = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
			rng_seed = rand(UInt32),
			sample_size = x
		) for _ in 1:5 for x in 100:200:2100
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
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=400,
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

# ╔═╡ 48be4adc-9751-4afc-8dd0-262f65895abb
sample_result = run(ensemble_config_sample_result, get_metrics, log_dir * "ensemble_sample_result.jld2", recalculate=false)

# ╔═╡ 7daf9dfc-f859-47f9-aeed-0e50054887cb
ensemble_config_sample_result.diffusion_run_configs[1].diffusion_steps

# ╔═╡ 0e665f5a-b8dc-487d-acb1-4aa66320f75d
sample_result_dff = filter(row -> row["diffusion_step"] ==  ensemble_config_sample_result.diffusion_run_configs[1].diffusion_steps, sample_result)

# ╔═╡ 905bddc3-f243-4e0a-9805-0bdfb0f3cb21
sample_result_dff[!, "sample_size"] = retrieve_variable(sample_result_dff, ["election_config", "sampling_config", "sample_size"])

# ╔═╡ 1d90fedd-0c41-4507-92e8-b84df5a59338
draw_voting_rule(agg_stats(sample_result_dff, "sample_size", "borda_scores"), "sample_size", "borda_scores")

# ╔═╡ 23c854aa-5040-4a04-8714-4986d8b46c62
md"Variability may be large because of DEG graph generation where in added nodes there may be a node with extremely high degree or not."

# ╔═╡ 8940150e-f07c-4f9d-93c8-5e872fb8bc36
md"#### Impact of Graph Type"

# ╔═╡ 81b233eb-ee65-4d28-b2fa-bc56e537be41
ensemble_config_graphType_result = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
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
			DEG_graph_config(
				rng_seed=rand(UInt32),
				target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
				target_cc=0.3,
				homophily=0.8
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
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=400,
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

# ╔═╡ acbfce7b-a7d1-4a75-9ccd-647dc05e3eab
graphType_result = run(ensemble_config_graphType_result, get_metrics, log_dir * "ensemble_graphType_result.jld2", recalculate=false)

# ╔═╡ 8aeade88-f959-4171-9951-c47faaf3f10e
begin
	graphType_result[!, "sample_size"] = retrieve_variable(graphType_result, ["election_config", "sampling_config", "sample_size"])
	graphType_result[!, "graph_config"] = retrieve_variable(graphType_result, ["model_config", "graph_config"])
end

# ╔═╡ 2717c36b-5c32-4cf2-a7b2-7e14b834de5b
graphType_result_dff = [
	filter(row -> typeof(row["graph_config"]) == DEG_graph_config && row["graph_config"].homophily==0.0, graphType_result),
	
	filter(row -> typeof(row["graph_config"]) == DEG_graph_config && row["graph_config"].homophily==0.8, graphType_result),
	
	filter(row -> typeof(row["graph_config"]) == BA_graph_config, graphType_result),
	filter(row -> typeof(row["graph_config"]) == Random_graph_config, graphType_result)
]

# ╔═╡ 10c76fd6-fb89-43b9-93ed-81ced694285f


# ╔═╡ ec952bc1-57fa-4f95-b5b9-67827c28bc53
compare_voting_rule(graphType_result_dff, "diffusion_step", "borda_scores", linestyles=linestyles, labels=["DEG, homophily=0", "DEG, homophily=0.8", "BA", "Random"])

# ╔═╡ aca1a017-ae15-4879-be84-7f2eb02779b7
compare_voting_rule(graphType_result_dff, "diffusion_step", "plurality_scores", linestyles=linestyles, labels=["DEG, homophily=0", "DEG, homophily=0.8", "BA", "Random"])

# ╔═╡ 0969d15f-216f-4c7f-ba18-59ef0e0ea542
md"
- group by setting
- for each candidate 
- get std of borda
"

# ╔═╡ 8c7c17b3-6b84-4e20-b9c5-84e18227f239
md"#### Impact of Activation Order"

# ╔═╡ 1019ae88-0044-4156-87d9-bd89ca11ab50
ensemble_config_diffusionSeed_result = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
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
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=400,
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

# ╔═╡ 42e0f5bb-b6ae-4c95-bdf6-bb91aa33787f
diffusionSeed_result = run(ensemble_config_diffusionSeed_result, get_metrics, log_dir * "ensemble_diffusionSeed_result.jld2", recalculate=false)

# ╔═╡ 452f9e35-6625-4c08-8988-8c857a17197d
compare_voting_rule([diffusionSeed_result], "diffusion_step", "borda_scores", linestyles=linestyles)

# ╔═╡ fdff4f5f-efad-4608-b4c3-8d5cf0f69fec
compare_voting_rule([diffusionSeed_result], "diffusion_step", "plurality_scores", linestyles=linestyles)

# ╔═╡ 3f1c264b-4aee-427f-8dfd-da5673072e0b
md"#### Impact of Influence Direction"

# ╔═╡ f889196d-d64e-4d63-93e2-569be2e4095e


# ╔═╡ 07689f7d-205b-4000-8673-d45e177815b4
md"#### Impact of Weights in SP"

# ╔═╡ 11d3e2b1-5c8a-4aea-b366-4427126d2e65
md"difference of mean, w/o vs with"

# ╔═╡ 5d86d9ee-9b05-486f-b2e6-4969c0653c54
ensemble_config_weighting_result = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
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
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=400,
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

# ╔═╡ 3b519c74-6abb-410b-a711-e10671c87685
weighting_result = run(ensemble_config_weighting_result, get_metrics, log_dir * "ensemble_weighting_result.jld2", recalculate=false)

# ╔═╡ 40702ce9-a22e-47e3-aaf2-dae71fe69e4f
weighting_result[!, "weighting_rate"] = retrieve_variable(weighting_result, ["model_config", "voter_config", "weighting_rate"])

# ╔═╡ d304cdf1-96c9-4bc9-ab4b-1f4fc1864267
compare_voting_rule(groupby(weighting_result, "weighting_rate"), "diffusion_step", "borda_scores", linestyles=linestyles)

# ╔═╡ ebd6e293-7818-4619-9e04-bc264284ee75
compare_voting_rule(groupby(weighting_result, "weighting_rate"), "diffusion_step", "plurality_scores", linestyles=linestyles)

# ╔═╡ 859706a0-7d68-4379-ad86-654296d29d7f
md"#### Impact of Stubbornness SP"

# ╔═╡ 093fc6d3-5d33-4ca3-9a00-c5d8cb0f430b
md"difference of mean, w/o vs with"

# ╔═╡ 7015811a-1d7e-4bee-80b8-b11e2c004b5a
ensemble_config_stubbornness_result_SP = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
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
			] for _ in 1:5
		];
		[
			[
				SP_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.2)
				)
			] for _ in 1:5
		];
		[
			[
				SP_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.3)
				)
			] for _ in 1:5
		]
	],
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=600,
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

# ╔═╡ a1cf685e-646d-4619-aecf-a483373babc2
stubbornness_result_SP = run(ensemble_config_stubbornness_result_SP, get_metrics, log_dir * "ensemble_stubbornness_result_SP.jld2", recalculate=false)

# ╔═╡ 321019e0-a8dc-483c-92fd-cb954d029cf4
stubbornness_result_SP[!, "stubbornness_std"] = std.(retrieve_variable(stubbornness_result_SP, ["diffusion_config", "diffusion_init_config", 1, "stubbornness_distr"]))

# ╔═╡ 62553a5a-cfa7-4dab-a2cb-ce90aa750d86
begin
	stubbornness_result_SP_gdf = groupby(stubbornness_result_SP, "stubbornness_std")
	stubbornness_result_SP_labels = ["Stubbornness std=" * string(key[1]) for (key, _) in pairs(stubbornness_result_SP_gdf)]
end

# ╔═╡ d7a1f736-0870-49d6-b2a9-d5767a21956e
[(key, val) for (key, val) in pairs(stubbornness_result_SP_gdf)]

# ╔═╡ fd198c81-9aad-4f2f-905d-583f927fd423
compare_voting_rule(stubbornness_result_SP_gdf, "diffusion_step", "borda_scores", linestyles=linestyles, labels=stubbornness_result_SP_labels)

# ╔═╡ f2f5e329-81ed-48cd-988e-48a96b1e5516
compare_voting_rule(stubbornness_result_SP_gdf, "diffusion_step", "plurality_scores", linestyles=linestyles, labels=stubbornness_result_SP_labels)

# ╔═╡ 6d097278-add3-4434-9602-2e3bf951468c
compare_metric(stubbornness_result_SP_gdf, "diffusion_step",  "unique_votes", linestyles=linestyles, labels=stubbornness_result_SP_labels)

# ╔═╡ b638c19a-b93d-45fa-985c-1a47c8f652ea
md"#### Impact of Stubbornness KT"

# ╔═╡ ff7f8855-4cea-47a7-99c0-2c75760e9e24
ensemble_config_stubbornness_result_KT = Ensemble_config(
	data_path = data_path,
	remove_candidate_ids= remove_candidate_ids,
	sampling_configs = [
		Sampling_config(
			rng_seed = rand(UInt32),
			sample_size = 1500
		)
	],
	
	voter_configs = [
		Kendall_voter_config(
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
				KT_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.1)
				)
			] for _ in 1:5
		];
		[
			[
				KT_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.2)
				)
			] for _ in 1:5
		];
		[
			[
				KT_mutation_init_config(
					rng_seed=rand(UInt32),
					stubbornness_distr=Distributions.Normal(0.5, 0.3)
				)
			] for _ in 1:5
		]
	],
	
	diffusion_run_configs = [
		Diffusion_run_config(
			diffusion_steps=600,
			mutation_configs=[
				KT_mutation_config(
					rng=Random.MersenneTwister(rand(UInt32)),
					evolve_vertices=1.0,
					attract_proba = 1.0
				)
			]
		)
	]
)

# ╔═╡ d863a58c-ee27-432f-b24b-2244ec27b430
stubbornness_result_KT = run(ensemble_config_stubbornness_result_KT, get_metrics, log_dir * "ensemble_stubbornness_result_KT.jld2", recalculate=false)

# ╔═╡ 6090fa1d-28dc-4c34-ba36-eeed2bd33b9e
stubbornness_result_KT[!, "stubbornness_std"] = std.(retrieve_variable(stubbornness_result_KT, ["diffusion_config", "diffusion_init_config", 1, "stubbornness_distr"]))

# ╔═╡ 0011a38e-a813-4a58-a224-4068e71151cc
begin
	stubbornness_result_KT_gdf = groupby(stubbornness_result_KT, "stubbornness_std")
	stubbornness_result_KT_labels = ["Stubbornness std=" * string(key[1]) for (key, _) in pairs(stubbornness_result_KT_gdf)]
end

# ╔═╡ 880577b3-88c5-41de-805b-d8b68ed2c99b
compare_voting_rule(stubbornness_result_KT_gdf, "diffusion_step", "borda_scores", linestyles=linestyles, labels=stubbornness_result_KT_labels)

# ╔═╡ 6cb35012-4a58-4e96-a21f-afd95821b59c
compare_voting_rule(stubbornness_result_KT_gdf, "diffusion_step", "plurality_scores", linestyles=linestyles, labels=stubbornness_result_KT_labels)

# ╔═╡ 43cc5c2c-b77a-4ac1-bc89-c6d5da3b7b54
compare_metric(stubbornness_result_KT_gdf, "diffusion_step",  "unique_votes", linestyles=linestyles, labels=stubbornness_result_KT_labels)

# ╔═╡ ac3e648a-1d70-408a-b151-6f5c099f404e
md"
- (Extract), (filter), (groupby), compare
- Extract, filter, draw metric
- Extract, filter, draw voting rule

1. extract from configs used variables or create a column consisting of the type of configs in the column
2. filter a subset of dataframe
3. create dataframe groups to compare

"

# ╔═╡ a210fc8f-5d85-464d-8b0b-3fba19579a56
md"## Diffusion with set seed"

# ╔═╡ 94c8e845-2069-4a49-b109-8c71f1ffa2cd
md"Identify interesting runs of diffusion"

# ╔═╡ 8e369b4d-bae6-4387-98d8-1ec15f78833c
metric = "borda_scores"

# ╔═╡ 1f3329dd-23ad-4376-a6c3-edf851bee771
can = 8

# ╔═╡ 3fe9215c-ba2f-4aaa-bcb0-1eb8f1982db8
md"""
---
**Execution barrier** $(@bind anal_run CheckBox())

---
"""

# ╔═╡ ebb18c60-75bf-45dd-8a20-3d402aed23ee
md"Load specific config and save all the logs for in depth analysis."

# ╔═╡ 1173f5f8-b355-468c-8bfb-beebff5ba12b
function extreme_runs(df, metric, can)
	values = [result[can] for result in df[!, metric]]
	
	return Dict(pairs(df[argmin(values), :])), Dict(pairs(df[argmax(values), :]))
end

# ╔═╡ 0cf5fa96-ff32-4853-b6c7-ef1843f5281f
function extreme_runs(df, metric)
	values = df[!, metric]
	return Dict(pairs(df[argmin(values), :])), Dict(pairs(df[argmax(values), :]))
end

# ╔═╡ a4f875d7-685e-43bb-a806-be6ae6547ffb
min_row, max_row = extreme_runs(KT_result, metric, can)
#min_row, max_row = extreme_runs(df, metric)

# ╔═╡ a35605c7-382c-4443-aa79-386b02834c40
if anal_run
	min_metrics = run_experiment(min_row[:experiment_config], experiment_name="min_borda_can_8", checkpoint=10)
	max_metrics = run_experiment(max_row[:experiment_config], experiment_name="max_borda_can_8", checkpoint=1)
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
# ╠═867bb4cb-7edc-4814-ad31-a7cfa403a289
# ╠═a02c3b7d-b69a-4f34-91ee-5a735340715d
# ╠═5faca0e5-7fef-42fd-a659-226dc74c0ad6
# ╟─51fc614b-5fae-4357-b3fe-537458060109
# ╠═6e796169-f1bc-4d7f-a23a-4f18a0d5a664
# ╟─b49566b8-e2bf-4e08-b655-ccc29309690a
# ╠═7ae9d4d7-925f-4cb0-9b37-c0369f3b32c2
# ╠═bd34a8b6-193c-4456-af08-c8462bff23ad
# ╠═85baf553-6584-4940-9c48-ed187e075705
# ╠═5a4937a0-abb0-440b-b9cc-ea43522f6674
# ╠═489639d0-31b0-4da7-8e16-e5d7a33fbd85
# ╠═13802e53-a71b-4a0e-9dbf-0320cb342512
# ╟─eabc778b-5af6-47d5-97ef-c6156786ef19
# ╟─978aca85-e644-40f3-ab0c-4dddf1c77b80
# ╠═30887bdf-9bcb-46ad-b93c-63ad0eb078f7
# ╠═f8492aca-23ea-406f-87a5-0722f4ee2330
# ╟─b3007e9c-7b02-40e7-8666-e485bbe6ab05
# ╠═4e06848c-4c0a-4441-9894-715f622ccf4e
# ╠═f4151588-8ea6-45e7-9c83-b8cd33fb820d
# ╟─9077caca-08cc-4371-91ab-c84e58815bf8
# ╠═90986999-59f8-4717-864e-20300d0d3918
# ╠═e544bb0d-7577-4ad6-a961-fd656fdd254b
# ╠═22ddc889-129d-4489-934d-740be4334c5c
# ╟─5745da71-6421-41d9-aa2f-1626a3892ea0
# ╟─48f5e7a0-260b-4153-a443-e4962a57861c
# ╠═a22ab6ed-3e64-41af-a2fc-b7c6a22a6c42
# ╟─1e6bb760-6a9c-4e6a-85b6-e2ec97653135
# ╠═5a7df6ee-e403-48ec-8349-0a29767d1af8
# ╟─a1c06a57-20d2-4f4e-a9e2-343a74ff7841
# ╠═9615b9f8-be60-4730-96e7-d6d45e68c346
# ╠═604d68b0-dff1-45f7-8a5b-ac2eb44949b2
# ╟─5837e6dd-122e-4d92-9e98-e38da9c96735
# ╟─03bd1c60-4389-4b52-90f4-d1ddf8ef9158
# ╠═6c98c5a0-b3c8-48dc-b848-caa7c6688493
# ╟─1cce7716-74eb-4766-a678-51bf46d588fd
# ╠═a5ca4eff-ccb7-4a48-b762-1d31343526a6
# ╟─a1cbb7fb-fb0a-46a4-a8f6-0bfed218cb3d
# ╠═37ff3a7d-1d8d-4e10-8d93-9ebc6b0b366a
# ╠═e1a67161-5e4c-4bc9-bb43-2ea60ed01e76
# ╠═30950dda-b87f-47c3-ad51-8da72d8c0b1a
# ╟─bacc2f88-a732-4c00-98f9-61c3b9eb2531
# ╟─b642ebb6-469b-43e3-aa77-b88ab419126f
# ╠═9acf1a32-26e9-49c3-bb8c-9c281a3840d2
# ╠═48be4adc-9751-4afc-8dd0-262f65895abb
# ╠═7daf9dfc-f859-47f9-aeed-0e50054887cb
# ╠═0e665f5a-b8dc-487d-acb1-4aa66320f75d
# ╠═905bddc3-f243-4e0a-9805-0bdfb0f3cb21
# ╠═1d90fedd-0c41-4507-92e8-b84df5a59338
# ╠═23c854aa-5040-4a04-8714-4986d8b46c62
# ╟─8940150e-f07c-4f9d-93c8-5e872fb8bc36
# ╠═81b233eb-ee65-4d28-b2fa-bc56e537be41
# ╠═acbfce7b-a7d1-4a75-9ccd-647dc05e3eab
# ╠═8aeade88-f959-4171-9951-c47faaf3f10e
# ╠═2717c36b-5c32-4cf2-a7b2-7e14b834de5b
# ╠═10c76fd6-fb89-43b9-93ed-81ced694285f
# ╠═ec952bc1-57fa-4f95-b5b9-67827c28bc53
# ╠═aca1a017-ae15-4879-be84-7f2eb02779b7
# ╠═0969d15f-216f-4c7f-ba18-59ef0e0ea542
# ╟─8c7c17b3-6b84-4e20-b9c5-84e18227f239
# ╠═1019ae88-0044-4156-87d9-bd89ca11ab50
# ╠═42e0f5bb-b6ae-4c95-bdf6-bb91aa33787f
# ╠═452f9e35-6625-4c08-8988-8c857a17197d
# ╠═fdff4f5f-efad-4608-b4c3-8d5cf0f69fec
# ╟─3f1c264b-4aee-427f-8dfd-da5673072e0b
# ╠═f889196d-d64e-4d63-93e2-569be2e4095e
# ╠═07689f7d-205b-4000-8673-d45e177815b4
# ╟─11d3e2b1-5c8a-4aea-b366-4427126d2e65
# ╠═5d86d9ee-9b05-486f-b2e6-4969c0653c54
# ╠═3b519c74-6abb-410b-a711-e10671c87685
# ╠═40702ce9-a22e-47e3-aaf2-dae71fe69e4f
# ╠═d304cdf1-96c9-4bc9-ab4b-1f4fc1864267
# ╠═ebd6e293-7818-4619-9e04-bc264284ee75
# ╟─859706a0-7d68-4379-ad86-654296d29d7f
# ╟─093fc6d3-5d33-4ca3-9a00-c5d8cb0f430b
# ╠═7015811a-1d7e-4bee-80b8-b11e2c004b5a
# ╠═a1cf685e-646d-4619-aecf-a483373babc2
# ╠═321019e0-a8dc-483c-92fd-cb954d029cf4
# ╠═d7a1f736-0870-49d6-b2a9-d5767a21956e
# ╠═62553a5a-cfa7-4dab-a2cb-ce90aa750d86
# ╠═fd198c81-9aad-4f2f-905d-583f927fd423
# ╠═f2f5e329-81ed-48cd-988e-48a96b1e5516
# ╟─6d097278-add3-4434-9602-2e3bf951468c
# ╟─b638c19a-b93d-45fa-985c-1a47c8f652ea
# ╠═ff7f8855-4cea-47a7-99c0-2c75760e9e24
# ╠═d863a58c-ee27-432f-b24b-2244ec27b430
# ╠═6090fa1d-28dc-4c34-ba36-eeed2bd33b9e
# ╠═0011a38e-a813-4a58-a224-4068e71151cc
# ╟─880577b3-88c5-41de-805b-d8b68ed2c99b
# ╟─6cb35012-4a58-4e96-a21f-afd95821b59c
# ╟─43cc5c2c-b77a-4ac1-bc89-c6d5da3b7b54
# ╠═ac3e648a-1d70-408a-b151-6f5c099f404e
# ╟─a210fc8f-5d85-464d-8b0b-3fba19579a56
# ╟─94c8e845-2069-4a49-b109-8c71f1ffa2cd
# ╠═8e369b4d-bae6-4387-98d8-1ec15f78833c
# ╠═1f3329dd-23ad-4376-a6c3-edf851bee771
# ╠═a4f875d7-685e-43bb-a806-be6ae6547ffb
# ╟─3fe9215c-ba2f-4aaa-bcb0-1eb8f1982db8
# ╟─ebb18c60-75bf-45dd-8a20-3d402aed23ee
# ╠═a35605c7-382c-4443-aa79-386b02834c40
# ╠═1173f5f8-b355-468c-8bfb-beebff5ba12b
# ╠═0cf5fa96-ff32-4853-b6c7-ef1843f5281f
