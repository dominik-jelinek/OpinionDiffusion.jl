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

# ╔═╡ 3852d8ef-4cd4-44b9-80b0-122afee6c73d
using DataFrames, Statistics

# ╔═╡ 419d618c-1ce4-4702-b2b0-9c3d160c245e
using GraphMakie, CairoMakie, Colors

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
import Distributions, Distances, Graphs, Plots, Random

# ╔═╡ 55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
md"Choose visualization backend from Plots library. Plotly for interactivity. GR for speed."

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
Plots.plotly()
#Plots.gr()

# ╔═╡ 46e8650e-f57d-48d7-89de-1c72e12dea45
md"## Election Stage"

# ╔═╡ bcc5468c-2a49-409d-b810-05fc30f4edca
md"""Select dataset: $(@bind input_filename Select([filename for filename in readdir("./data") if split(filename, ".")[end] == "toc"], default = "ED-00001-00000002.toc"))"""

# ╔═╡ 0d325100-a1b1-4838-b407-017cef8e4456
data_path = "./data/" * input_filename

# ╔═╡ c6ccf2a8-e045-4da9-bbdb-270327c2d53f
init_election = parse_data(data_path)

# ╔═╡ c8522415-1fd9-4c06-bf2a-38ab23153b56
md"### Dataset Summary"

# ╔═╡ 61fa4389-3b1e-4be3-a5a2-ee37218d7d55
draw_election_summary_frequencies(get_election_summary(get_votes(init_election), drop_last=true))

# ╔═╡ 977d39e2-7f82-49e8-a93f-889204bd19cb
md"### Remove unnecessary candidates"

# ╔═╡ 8ea22c93-1fe3-44b2-88c1-fb6ccd195866
if input_filename == "ED-00001-00000001.toc"
	remove_candidate_ids = [3, 5, 8, 11]
elseif input_filename == "ED-00001-00000002.toc"
	remove_candidate_ids = [8]#[1, 6, 8]
elseif input_filename == "ED-00001-00000003.toc"
	remove_candidate_ids = [3, 8, 9, 11]
else
	remove_candidate_ids = []
end

# ╔═╡ 4aa4cdb4-af2a-4fcb-913d-389a9a899ab0
election_removed_candidates = remove_candidates(init_election, remove_candidate_ids)

# ╔═╡ 100d0645-1178-428a-b4c1-3859ecb3ee18
md"### Sample voters"

# ╔═╡ 1626bf3e-8fe6-4e21-8cf4-76d84848114f
sampling_config = Sampling_config(
	rng_seed=69,
	sample_size=min(1000, length(get_votes(election_removed_candidates)))
)

# ╔═╡ 8a760833-139b-406b-b985-a1daf7585ed3
election = sample(election_removed_candidates, sampling_config)

# ╔═╡ 24f7f4d4-301f-4fd7-a459-e14fe5d5aaee
candidates = get_candidates(election)

# ╔═╡ 20893df2-fa42-46bf-889e-582b9ac39164
md"## Model Stage"

# ╔═╡ 831816fa-7983-4c05-9a76-a03631bf1e57
md"### Voter config"

# ╔═╡ 40fdc182-c693-48fb-99ee-43d1bc78d95f
md"#### Weighting of Spearman voter"

# ╔═╡ aad327bc-5554-4ba2-aa7a-1eadadce10c0
weighting_rate = 1.0

# ╔═╡ e41a36e8-418a-485c-94e6-8f77edd2a315
weightss, eps = OpinionDiffusion.spearman_weights(weighting_rate, length(candidates))

# ╔═╡ 3a0442fa-0db8-49fe-9520-9edb98c7d68f
Plots.bar([string(i) * '-' * string(i + 1) for i in 1:length(candidates)-1], [abs(weightss[x] - weightss[y]) for (x,y) in zip(1:length(candidates)-1, 2:length(candidates))], legend=false, ylabel="Distance Between Buckets", xlabel="Bucket Positions")

# ╔═╡ 4845cd62-971c-4420-b8a5-2bb3d741bd8f
eps

# ╔═╡ ed315e83-6d73-4f9a-afb9-f0174e08ef29
md"#### Voter selection"

# ╔═╡ a87ea61d-13d9-4c91-b08e-9f24cde3d290
md"""Select voter type: $(@bind voter_type Select(["Kendall-tau voter", "Spearman voter"], default="Spearman voter"))"""

# ╔═╡ 10cb247f-d445-4091-b863-49deeb4c35fe
if voter_type == "Spearman voter"
	voter_config = Spearman_voter_config(
		weighting_rate = weighting_rate
	)
else# voter_type == "Kendall-tau voter"
	voter_config = Kendall_voter_init_config(
	)
end

# ╔═╡ d1de5e6d-cbfd-40af-9580-f5633105c63e
voters = init_voters(get_votes(election), voter_config)

# ╔═╡ c91395e7-393a-4ee1-ab44-7269cb1314d8
md"### Graph config"

# ╔═╡ 48a2cf2b-bd86-4131-a230-290124cc5f48
md"#### DEG target degree distribution"

# ╔═╡ f012e77d-2703-4922-aa3e-7dd6f04757e4
begin
	exp = 0.7
	scale = 2.0
	max_degree = 500
end

# ╔═╡ c4dfe306-aad8-4248-bc9e-c2de841a7354
md"#### Select a method for graph generation"

# ╔═╡ 738b7617-00c5-4d25-ae1b-61788ba23f5c
homophily = 0.8

# ╔═╡ 39256cbd-7807-42bc-81b1-d6f2128ccaf9
md"""Select graph generation method: $(@bind graph_type Select(["DEG", "Barabasi-Albert", "Random"]))"""

# ╔═╡ 7e0d083d-2de1-4a4c-8d19-7dea4f95152a
if graph_type == "DEG"
	graph_config = DEG_graph_config(
		rng_seed=rand(UInt32),
		target_deg_distr=Distributions.truncated(Distributions.Pareto(0.7, 2.0); upper=500),
		target_cc=0.3,
		homophily=0.0
	)
elseif graph_type == "BA"
	graph_config = BA_graph_config(
		rng_seed=rand(UInt32),
		average_degree=18,
		homophily=0.0
	)
else
	graph_config = Random_graph_config(
		rng_seed=rand(UInt32),
		average_degree=18
	)
end

# ╔═╡ 00c4a4be-33a7-4943-9f03-ecbb71d2a2b2
social_network = init_graph(voters, graph_config)				

# ╔═╡ 3cede6ac-5765-4c72-9b53-103c9c6a9bd9
md"### Create model"

# ╔═╡ 341a601f-2727-4362-b7d1-29dd47a92539
md"#### Model config"

# ╔═╡ 98885ec6-7561-43d7-bdf6-7f58fb2720f6
md"Choose source of the model and then check execution barrier for generation of the model"

# ╔═╡ 9b6e56ed-cd89-4684-83d7-b2f931a07d14
model_config = General_model_config(
	voter_config=voter_config,
	graph_config=graph_config
)

# ╔═╡ 1937642e-7f63-4ffa-b01f-22208a716dac
md"""
---
**Execution barrier** $(@bind cb_model CheckBox())

---
"""

# ╔═╡ 6518d9c8-4a21-4a9d-881c-bf4e6719f2df
if cb_model
	model = init_model(election, model_config)
end

# ╔═╡ 2c837f03-3329-4ab3-ad31-b0fe79df6bb7
md"## Initial Model Analysis"

# ╔═╡ a6590191-0fbd-41a9-a7c3-b99e68bb27aa
begin
	sample_size = length(model.voters)
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)
end

# ╔═╡ a652cf9b-adb5-47d2-906f-a7b479face45
vs = timestamp_vis(model, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ 5bf9306d-ed0f-4d12-af1f-75ba9fe1e9a4
vs[1][2]

# ╔═╡ 4f7384d8-4c16-4a75-93ca-755215d74643
begin
	f = Figure()
	OpinionDiffusion.draw_degree_cc!(f[1,1], get_social_network(model))
	f
end

# ╔═╡ 09540198-bead-4df1-99bd-6e7848e7d215
md"## Diffusion Stage"

# ╔═╡ 6f40b5c4-1252-472c-8932-11a2ee0935d2
md"Setup diffusion parameters and then check execution barrier for confirmation."

# ╔═╡ 9a2c4ed1-564e-4ab5-82f1-5b6c0c2b7cd0
md"### Configure Diffusion"

# ╔═╡ e3f2c391-4fc3-47b5-bad2-5abc8f89a345
if voter_type == "Spearman voter"
	init_voter_mutation_config = SP_mutation_init_config(
		rng_seed=rand(UInt32),
		stubbornness_distr=Distributions.Normal(0.5, 0.0)
	)
	
	voter_mutation_config = SP_mutation_config(
		rng=Random.MersenneTwister(rand(UInt32)),
		evolve_vertices=1.0,
		attract_proba = 1.0,
		change_rate = 0.05,
		normalize_shifts = true
	)
else #Kendall voter
	init_voter_mutation_config = KT_mutation_init_config(
		rng_seed=static_seed,
		stubbornness_distr=Distributions.Normal(0.5, 0.0)
	)
	
	voter_mutation_config = KT_mutation_config(
		rng=Random.MersenneTwister(rand(UInt32)),
		evolve_vertices=1.0,
		attract_proba = 1.0
	)
end

# ╔═╡ e7f02f43-f8d5-4b7d-b097-4cbe7c7541a7
graph_mutation_config = Graph_mutation_config(
	rng=Random.MersenneTwister(rand(UInt32)),
	evolve_edges=0.0,
	homophily = homophily
)

# ╔═╡ a815d8eb-7a4f-4509-923c-af6425d43619
diffusion_config = Diffusion_config(
	diffusion_init_configs = [
		init_voter_mutation_config
	],
	diffusion_run_config = Diffusion_run_config(
		diffusion_steps=400,
		mutation_configs=[
			voter_mutation_config, 
			graph_mutation_config
		]
	)
)

# ╔═╡ de772425-25de-4228-b12e-d567b8ceb20f
md"### Run Diffusion"

# ╔═╡ f39c07e5-ea47-4523-be1c-0ceeb8e99ba0
begin
	log_dir = "./logs"
	experiment_name = "experiment"
	interval = 1
end

# ╔═╡ 8e0750a9-2b83-4652-805b-dc1be2484161
md"""
Experiment logger $(@bind logging CheckBox(default=true))
"""

# ╔═╡ 274f8f7f-3f7d-4d71-b20b-20b5585dfcf7
function get_metrics(model)
	g = get_social_network(model)
    voters = get_voters(model)
	candidates = get_candidates(model)
	can_count = length(candidates)
	
	histogram = Graphs.degree_histogram(g)
    keyss = collect(keys(histogram))
    
	votes = get_votes(voters)

	metrics = Dict(
		#"min_degrees" => minimum(keyss),
        #"avg_degrees" => Graphs.ne(g) * 2 / Graphs.nv(g),
        #"max_degrees" => maximum(keyss),
        "avg_edge_dist" => mean(get_edge_distances(g, voters)),
        #"clustering_coefficient" => Graphs.global_clustering_coefficient(g), # Expensive
        #"diameter" => Graphs.diameter(g), #EXPENSIVE
        
        "avg_vote_length" => mean([length(vote) for vote in votes]),
        "unique_votes" => length(unique(votes)),
        
        "plurality_scores" => plurality_voting(votes, can_count, true),
        "borda_scores" => borda_voting(votes, can_count, true),
        #"copeland_scores" => copeland_voting(votes, can_count),
	)
	
	return metrics
end

# ╔═╡ 90b09d11-5368-4598-9f89-aaf3945206f6
md"""
Calculate metrics $(@bind metrics CheckBox(default=true))
"""

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
---
**Execution barrier** $(@bind cb_run CheckBox())

---
"""

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_run
	if logging
		experiment_logger = Experiment_logger(
			log_dir=log_dir, 
			experiment_name=experiment_name, 
			interval=interval
		)
		
		experiment_config = Experiment_config(
			election_config=Election_config(
				data_path=data_path, 			remove_candidate_ids=remove_candidate_ids, 		sampling_config=sampling_config
			),
			model_config=model_config,
			diffusion_config=diffusion_config
		)
		init_experiment(experiment_logger, model, experiment_config)
	else
		experiment_logger = nothing
	end

	if metrics
		accumulator = Accumulator(get_metrics)
		add_metrics!(accumulator, model)
	else
		accumulator = nothing
	end
	
	diffused_model, actions = diffusion(model, diffusion_config, experiment_logger=experiment_logger, accumulator=accumulator)
end

# ╔═╡ aad0da61-59d2-4429-97fa-6612872bb863
md"## Diffusion Analysis"

# ╔═╡ d716423e-7945-4e0a-a6ab-17e0b94c721e
md"### Compounded metrics"

# ╔═╡ 989fba90-fc00-4b23-b77d-88d738f5aab3
accumulated_metrics(accumulator)

# ╔═╡ 4c140c0e-71c7-4567-b37e-286395a450a3
md"
### What statistics do we care about during diffusion:
Graph dynamics:
- min degree
- avg degree
- max degree
- diameter
- global clustering coefficient

Vote dynamics:
- avg neighbor distance
- unique votes
- avg vote length

Voting rules:
- plurality
- borda
- copeland"

# ╔═╡ f539cf71-34ae-4e22-a7ac-d259b55cb2d3
md"
### What visualizations are important to understand specific state of society:
Voter centric: (graph, voters)
- ego network
- voter centralities
- voter properties to centralities correlation matrix? (which properties of the voter are important for centralities)

Communities: (graph, clusters)
- community graph
- avg distances between communities
- density of communities
- clustering coefficient
- centroid
- number of connections in between communities

Graph centric: (graph)
- Degree distribution
- Edge distance distribution
"

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
# ╠═3852d8ef-4cd4-44b9-80b0-122afee6c73d
# ╟─55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
# ╠═ff873978-d93f-4ba2-aadd-6cfd3b136e3d
# ╠═419d618c-1ce4-4702-b2b0-9c3d160c245e
# ╟─46e8650e-f57d-48d7-89de-1c72e12dea45
# ╟─bcc5468c-2a49-409d-b810-05fc30f4edca
# ╠═0d325100-a1b1-4838-b407-017cef8e4456
# ╠═c6ccf2a8-e045-4da9-bbdb-270327c2d53f
# ╟─c8522415-1fd9-4c06-bf2a-38ab23153b56
# ╠═61fa4389-3b1e-4be3-a5a2-ee37218d7d55
# ╟─977d39e2-7f82-49e8-a93f-889204bd19cb
# ╠═8ea22c93-1fe3-44b2-88c1-fb6ccd195866
# ╠═4aa4cdb4-af2a-4fcb-913d-389a9a899ab0
# ╟─100d0645-1178-428a-b4c1-3859ecb3ee18
# ╠═1626bf3e-8fe6-4e21-8cf4-76d84848114f
# ╠═8a760833-139b-406b-b985-a1daf7585ed3
# ╠═24f7f4d4-301f-4fd7-a459-e14fe5d5aaee
# ╟─20893df2-fa42-46bf-889e-582b9ac39164
# ╟─831816fa-7983-4c05-9a76-a03631bf1e57
# ╟─40fdc182-c693-48fb-99ee-43d1bc78d95f
# ╠═aad327bc-5554-4ba2-aa7a-1eadadce10c0
# ╠═e41a36e8-418a-485c-94e6-8f77edd2a315
# ╠═3a0442fa-0db8-49fe-9520-9edb98c7d68f
# ╠═4845cd62-971c-4420-b8a5-2bb3d741bd8f
# ╟─ed315e83-6d73-4f9a-afb9-f0174e08ef29
# ╟─a87ea61d-13d9-4c91-b08e-9f24cde3d290
# ╠═10cb247f-d445-4091-b863-49deeb4c35fe
# ╠═d1de5e6d-cbfd-40af-9580-f5633105c63e
# ╟─c91395e7-393a-4ee1-ab44-7269cb1314d8
# ╟─48a2cf2b-bd86-4131-a230-290124cc5f48
# ╠═f012e77d-2703-4922-aa3e-7dd6f04757e4
# ╟─c4dfe306-aad8-4248-bc9e-c2de841a7354
# ╠═738b7617-00c5-4d25-ae1b-61788ba23f5c
# ╟─39256cbd-7807-42bc-81b1-d6f2128ccaf9
# ╠═7e0d083d-2de1-4a4c-8d19-7dea4f95152a
# ╠═00c4a4be-33a7-4943-9f03-ecbb71d2a2b2
# ╟─3cede6ac-5765-4c72-9b53-103c9c6a9bd9
# ╟─341a601f-2727-4362-b7d1-29dd47a92539
# ╟─98885ec6-7561-43d7-bdf6-7f58fb2720f6
# ╠═9b6e56ed-cd89-4684-83d7-b2f931a07d14
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╠═6518d9c8-4a21-4a9d-881c-bf4e6719f2df
# ╟─2c837f03-3329-4ab3-ad31-b0fe79df6bb7
# ╠═a6590191-0fbd-41a9-a7c3-b99e68bb27aa
# ╠═a652cf9b-adb5-47d2-906f-a7b479face45
# ╠═5bf9306d-ed0f-4d12-af1f-75ba9fe1e9a4
# ╠═4f7384d8-4c16-4a75-93ca-755215d74643
# ╠═09540198-bead-4df1-99bd-6e7848e7d215
# ╟─6f40b5c4-1252-472c-8932-11a2ee0935d2
# ╟─9a2c4ed1-564e-4ab5-82f1-5b6c0c2b7cd0
# ╠═e3f2c391-4fc3-47b5-bad2-5abc8f89a345
# ╠═e7f02f43-f8d5-4b7d-b097-4cbe7c7541a7
# ╠═a815d8eb-7a4f-4509-923c-af6425d43619
# ╟─de772425-25de-4228-b12e-d567b8ceb20f
# ╠═f39c07e5-ea47-4523-be1c-0ceeb8e99ba0
# ╠═8e0750a9-2b83-4652-805b-dc1be2484161
# ╠═274f8f7f-3f7d-4d71-b20b-20b5585dfcf7
# ╠═90b09d11-5368-4598-9f89-aaf3945206f6
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╟─aad0da61-59d2-4429-97fa-6612872bb863
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═989fba90-fc00-4b23-b77d-88d738f5aab3
# ╟─4c140c0e-71c7-4567-b37e-286395a450a3
# ╟─f539cf71-34ae-4e22-a7ac-d259b55cb2d3
# ╠═187da043-ee48-420a-91c7-5e3a4fdc30bb
# ╠═ff89eead-5bf5-4d52-9134-aa415ae156b7
# ╠═e70e36ad-066e-4619-a06a-56e325745a0e
# ╠═b94c9f37-b321-4db2-9da9-75917be8e52e
