### A Pluto.jl notebook ###
# v0.19.9

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

# ╔═╡ 0009fe9c-117f-433c-b65f-2a0835aca0b9
using DataFrames

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

# ╔═╡ 75588cfd-52c9-4406-975c-c03158db6e78
import Distributions, Distances, Graphs, Plots, Random

# ╔═╡ 55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
md"Choose visualization backend from Plots library. Plotly for interactivity. GR for speed."

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
Plots.plotly()
#Plots.gr()

# ╔═╡ 46e8650e-f57d-48d7-89de-1c72e12dea45
md"## Load dataset"

# ╔═╡ bcc5468c-2a49-409d-b810-05fc30f4edca
md"""Select dataset: $(@bind input_filename Select([filename for filename in readdir("./data") if split(filename, ".")[end] == "toc"], default = "ED-00001-00000002.toc"))"""

# ╔═╡ c6ccf2a8-e045-4da9-bbdb-270327c2d53f
parties, src_candidates, src_election = parse_data(input_filename)

# ╔═╡ c8522415-1fd9-4c06-bf2a-38ab23153b56
md"## Explore dataset"

# ╔═╡ 21f341c3-4d38-4449-b0f8-62e4ab14c99b
strict_orders = [length(vote[end]) > 1 ? vote[1:end - 1] : vote  for vote in src_election ]

# ╔═╡ 042ec90d-8c0b-4acb-8d4c-31f701876cb6
function get_counts(votes, can_count)
	result = zeros(Int64, can_count, can_count)
	
	for vote in votes
        for (i, bucket) in enumerate(vote)
			#iterate buckets in vote
            result[iterate(bucket)[1], i] += 1
        end
    end
	
	return result
end

# ╔═╡ 06d301e2-90c1-4fdf-b5ab-e127f8dee777
counts = get_counts(strict_orders, length(src_candidates))

# ╔═╡ d5d394c0-9b7c-4255-95fd-dc8cc32ca018
Plots.heatmap(counts, yticks=1:length(src_candidates), xticks=1:length(src_candidates), ylabel="Candidate ID", xlabel="Position", title="Voting matrix")#[parties[can.party] for can in candidates])

# ╔═╡ 20078431-5a7c-4a6f-b181-984ed54cf506
Plots.bar(sum(counts, dims = 2), legend=false, xticks=1:length(src_candidates), ylabel="# votes", xlabel="Candidate ID", title="Candidate frequency")

# ╔═╡ 8c709e84-66f8-4128-a85c-66a4c5ffc9b7
Plots.bar(transpose(sum(counts, dims = 1)), legend=false, xticks=1:length(src_candidates), ylabel="# votes", xlabel="Vote position", title="Position frequency")

# ╔═╡ 977d39e2-7f82-49e8-a93f-889204bd19cb
md"## Sample voters and remove unnecessary candidates"

# ╔═╡ 8ea22c93-1fe3-44b2-88c1-fb6ccd195866
if input_filename == "ED-00001-00000001.toc"
	remove_candidates = [3, 5, 8, 11]
elseif input_filename == "ED-00001-00000002.toc"
	remove_candidates = [8]#[1, 6, 8]
elseif input_filename == "ED-00001-00000003.toc"
	remove_candidates = [3, 8, 9, 11]
else
	remove_candidates = []
end

# ╔═╡ 28f103a9-2b18-4cfc-bcf3-34d512f8da03
filtered_election, candidates = OpinionDiffusion.filter_candidates(src_election, src_candidates, remove_candidates, length(src_candidates))

# ╔═╡ ad94415f-fc56-4a2f-9068-e05d8633aafe
init_sample_size = min(1000, length(filtered_election))

# ╔═╡ 93b8909b-479c-422a-b475-2befadf5e9ec
election = filtered_election[OpinionDiffusion.StatsBase.sample(1:length(filtered_election), init_sample_size, replace=false)]

# ╔═╡ 20893df2-fa42-46bf-889e-582b9ac39164
md"## Configure model"

# ╔═╡ 831816fa-7983-4c05-9a76-a03631bf1e57
md"### Voter config"

# ╔═╡ 40fdc182-c693-48fb-99ee-43d1bc78d95f
md"#### Weighting of Spearman voter"

# ╔═╡ a38a2e2e-e742-4c0b-9cf5-cd8376178300
weighting_rate = 0.0

# ╔═╡ 342d3d34-4a3a-4f2d-811b-e9ab143504fd
begin
	weight_func = position -> (length(candidates) - position)^weighting_rate
	weights = Vector{Float64}(undef, length(candidates))
	weights[1] = 0.0
	for i in 2:length(weights)
		weights[i] = weights[i - 1] + weight_func(i - 1)
	end
	max_sp_distance = OpinionDiffusion.get_max_distance(length(candidates), weights)
	weights = weights ./ max_sp_distance
end

# ╔═╡ c2d75170-789d-41b4-b30b-40b143d23702
max_sp_distance

# ╔═╡ 825b45d1-fe6a-4460-8fe2-b323019c56b6
Plots.bar([string(i) * '-' * string(i + 1) for i in 1:length(candidates)-1], [weight_func(x) for x in 1:length(candidates)-1], legend=false)

# ╔═╡ ed315e83-6d73-4f9a-afb9-f0174e08ef29
md"#### Voter selection"

# ╔═╡ a87ea61d-13d9-4c91-b08e-9f24cde3d290
md"""Select voter type: $(@bind voter_type Select(["Kendall-tau voter", "Spearman voter"], default="Spearman voter"))"""

# ╔═╡ 10cb247f-d445-4091-b863-49deeb4c35fe
if voter_type == "Spearman voter"
	voter_init_config = Spearman_voter_init_config(
		weights = weights, 
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)
else# voter_type == "Kendall-tau voter"
	voter_init_config = Kendall_voter_init_config(
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)
end

# ╔═╡ c91395e7-393a-4ee1-ab44-7269cb1314d8
md"## Graph config"

# ╔═╡ 48a2cf2b-bd86-4131-a230-290124cc5f48
md"#### DEG target degree distribution"

# ╔═╡ c4dfe306-aad8-4248-bc9e-c2de841a7354
md"#### Select a method for graph generation"

# ╔═╡ 738b7617-00c5-4d25-ae1b-61788ba23f5c
homophily = 0.5

# ╔═╡ 39256cbd-7807-42bc-81b1-d6f2128ccaf9
md"""Select graph generation method: $(@bind graph_type Select(["DEG", "Barabasi-Albert"]))"""

# ╔═╡ 7e0d083d-2de1-4a4c-8d19-7dea4f95152a
if graph_type == "DEG"
	graph_init_config = DEG_graph_config(
										exp = 0.7,
										scale = 2.0,
										max_degree = min(500, floor(init_sample_size / 4)),
										target_cc = 0.3, 
										homophily = homophily
									)
else# graph_type == "BA"
	graph_init_config = BA_graph_config(
										m=10, 
										homophily=homophily
									)
end

# ╔═╡ 3b67247a-7dd8-49e7-ad88-64a8dad90d17
begin
	pareto = Distributions.truncated(Distributions.Pareto(graph_init_config.exp, graph_init_config.scale); upper=graph_init_config.max_degree)
	
	target_degree_distr = Int.(round.(rand(pareto, init_sample_size)))
end

# ╔═╡ 25374289-3cf1-4d27-887f-a47f9794f2bc
Plots.histogram(target_degree_distr)

# ╔═╡ 7069907d-2949-4f68-9602-9bef9ff46064
sum(target_degree_distr) / init_sample_size

# ╔═╡ 338aa155-a4fb-4ff9-b27d-fc5481abb58d
md"## Model config"

# ╔═╡ 0d3b58fe-8205-4d4c-9174-09cec37a61f3
model_config = General_model_config(
	voter_init_config = voter_init_config,
	graph_init_config = graph_init_config,
)

# ╔═╡ 759557c0-228f-4827-9038-1963b54a08d9
md"##### Locate model"

# ╔═╡ b4d60582-ab23-4b2e-84bb-efd60786dc93
[file for file in readdir("logs")]

# ╔═╡ 637a053e-1ce8-4107-926a-46f46b4718f8
log_dir = "logs"

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir = log_dir * "/" * "model_2022-07-20_10-55-06"

# ╔═╡ 37cf8471-02a7-4a91-b17e-f44d06974d97
[file for file in readdir(model_dir)]

# ╔═╡ d8b613c5-7276-466d-a54a-7670f0921b35
exp_dir = model_dir * "/" * "experiment_2022-07-20_10-55-06"

# ╔═╡ a35b87d8-76eb-4b3e-b195-19dd5564bcf8
[exp_dir * "/" * file for file in readdir(exp_dir)]

# ╔═╡ 4712586c-bc93-43df-ae66-1e75a21b6f85
md"Index for loading specific model state inside of exp_dir. Insert -1 for the last state of the model."

# ╔═╡ 571e7a33-20b7-4432-b553-c07b9081c68d
idx = -1

# ╔═╡ 3cede6ac-5765-4c72-9b53-103c9c6a9bd9
md"## Create/load model"

# ╔═╡ 98885ec6-7561-43d7-bdf6-7f58fb2720f6
md"Choose source of the model and then check execution barrier for generation of the model"

# ╔═╡ 72450aaf-c6c4-458e-9555-39c31345116b
md"""
Model source $(@bind model_source Select(["restart_model" => "Restart", "load_model" => "Load", "new_model" => "New (takes time)"]))
"""

# ╔═╡ 8e0750a9-2b83-4652-805b-dc1be2484161
md"""
Logging $(@bind logging CheckBox())
"""

# ╔═╡ 1937642e-7f63-4ffa-b01f-22208a716dac
md"""
---
Execution barrier $(@bind cb_model CheckBox())

---
"""

# ╔═╡ fa47c89c-784f-4552-9e34-22e499a0231f
model_seed = rand(UInt32)

# ╔═╡ 93cb669f-6743-4b50-80de-c8594ea20497
if cb_model && logging
	if model_source == "new_model"
		model = init_model(election, length(candidates), model_config, model_seed)
		logger = Logger(model)
		
	elseif model_source == "load_model" # load specific state and start new experiment
		model, logger = load_model(model_dir, exp_dir, idx, true)
		
	else # restart and create new experiment
		model, logger = restart_model(model_dir)
	end
elseif cb_model && !logging
	if model_source == "new_model"
		model = init_model(election, length(candidates), model_config, model_seed)
		
	elseif model_source == "load_model"
		model = load_log(exp_dir, idx)
		
	else # restart
		model = load_log(model_dir)
	end
	logger = nothing
end

# ╔═╡ 776f99ea-6e44-4eb6-b2dd-3552bbb39954
logger.model_dir, logger.exp_dir

# ╔═╡ 2c837f03-3329-4ab3-ad31-b0fe79df6bb7
md"## Initial metrics"

# ╔═╡ 29cf96fa-4b08-4909-a70e-288441da15ad
sort(Graphs.degree(model.social_network))

# ╔═╡ 9609ec4b-ed07-4121-8db3-f7d374698ac0
sort(OpinionDiffusion.get_edge_distances2(model.social_network, model.voters), by=x -> x[3], rev=true)

# ╔═╡ 923cde07-7b5a-46a8-86a8-6c076c61631b
begin
	a = 208
	b = 987
end

# ╔═╡ 3bea0fad-f106-49f4-be64-2ba7361591a6
get_vote(model.voters[a])

# ╔═╡ 6f5eb499-b01d-4e05-a380-04ac17345b68
get_opinion(model.voters[a])

# ╔═╡ 05ad5115-2a2b-479e-96d1-9796572d2f54
get_vote(model.voters[b])

# ╔═╡ 07893eca-c6ea-411a-94f5-802134c46311
get_opinion(model.voters[b])

# ╔═╡ 868709ae-5084-42db-a3c4-94fe2180bb05
length(unique(get_votes(get_voters(model))))

# ╔═╡ 045b7c19-a26d-4237-8a50-55abf7f9b57c
length(Graphs.neighbors(model.social_network, 1))

# ╔═╡ 09540198-bead-4df1-99bd-6e7848e7d215
md"## Setup diffusion"

# ╔═╡ 6f40b5c4-1252-472c-8932-11a2ee0935d2
md"Setup diffusion parameters and then check execution barrier for confirmation."

# ╔═╡ b6cd7a31-80b9-49eb-8004-34de5e6ad910
attract_proba = 1.0

# ╔═╡ 0295587e-ad65-4bf1-b6d4-1b87a1e844ff
voter_diff_config_sp = Spearman_voter_diff_config(
			attract_proba = attract_proba,
			change_rate = 0.1,
			normalize_shifts = (true, weights[1], weights[end])
        )

# ╔═╡ 63a4233d-c596-4844-8539-91a2223f2266
voter_diff_config_kt = Kendall_voter_diff_config(attract_proba = attract_proba)

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diffusion_config = Diffusion_config(
        checkpoint = 10, #only for a run without ensembling
		evolve_vertices = 1.0,
		evolve_edges = 0.0,
        voter_diff_config = voter_init_config isa Spearman_voter_init_config ? voter_diff_config_sp : voter_diff_config_kt,
        graph_diff_config = General_graph_diff_config(
            homophily = homophily
        )
    )

# ╔═╡ d877c5d0-89af-48b9-bcd0-c1602d58339f
diffusions = 100

# ╔═╡ 27a60724-5d19-419f-b208-ffa0c78e2505
ensemble_size = 5

# ╔═╡ 87c573c1-69a4-4a61-bbb8-acb716f8ec6d
ensemble_model = true

# ╔═╡ de772425-25de-4228-b12e-d567b8ceb20f
md"## Run diffusion"

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
---
Execution barrier $(@bind cb_run CheckBox())

---
"""

# ╔═╡ f2ca9891-3663-4ec6-bc69-e454faabba53
reduce(.+, dfs) ./ length(dfs)

# ╔═╡ 3fe9215c-ba2f-4aaa-bcb0-1eb8f1982db8
md"""
---
Execution barrier $(@bind anal_run CheckBox())

---
"""

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

# ╔═╡ 56d7f1d9-f291-446b-8281-013d017102f2
#model_metrics = ensemble_init_model_AB(3, election, length(candidates), init_metrics, update_metrics!, model_config, diffusion_config)

# ╔═╡ ccbfad20-549e-4275-8bbb-644157d98926
pops = collect(0.0:0.5:1.0)

# ╔═╡ 409b3357-dcf0-4ddb-8a4b-b7e2f21cee8f
#model_metrics = ensemble_init_model(false, 3, election, length(candidates), init_metrics, update_metrics!, diffusion_config)

# ╔═╡ 19c42165-d4ba-4c86-b134-514f6b017fe9
#model_metrics = ensemble_init_model_DEG(3, election, length(candidates), init_metrics, update_metrics!, model_config, diffusion_config, 0.0)

# ╔═╡ 13624365-3fba-49c7-bbe0-b33f9865e953
model_metrics

# ╔═╡ f560c9f3-1b7d-48c0-b8d7-40f4b43d3cec
sizes = collect(100:100:length(election))

# ╔═╡ 8fd45292-6704-4b57-b7e4-b650dce8d19c

#p = OpinionDiffusion.draw_range([x[2] for x in ccs], [x[3] for x in ccs], [x[4] for x in ccs], label="", x = sizess)

# ╔═╡ a1714045-adf8-427e-a960-dad1fda7aaa3
#=begin
	p = Plots.plot(title="Barabasi-Albert graph (m=5)", xlabel="sample size", ylabel="clustering coefficient", legend=true)
	for (i, pop) in enumerate(pops)
		ccs = model_metrics[i]["clustering_coeff"]
		OpinionDiffusion.draw_range!(p, [x[2] for x in ccs], [x[3] for x in ccs], [x[4] for x in ccs], label="Homophily: " * string(pop),c=i, x = sizes)
	end
	Plots.savefig(p, "img/AB_size_cc_hom.pdf")
	p
end=#

# ╔═╡ 2d3e4761-d868-42ee-99fa-40f2a9ee522b
begin
	p = Plots.plot(title="DEG graph (target_cc=0.5)", xlabel="sample size", ylabel="clustering coefficient", legend=true)
	for (i, pop) in enumerate(pops)
		ccs = model_metrics[i]["clustering_coeff"]
		OpinionDiffusion.draw_range!(p, [x[2] for x in ccs], [x[3] for x in ccs], [x[4] for x in ccs], label="Homophily: " * string(pop),c=i, x = sizes)
	end
	Plots.savefig(p, "img/DEG_size_cc05_hom.pdf")
	p
end

# ╔═╡ aad0da61-59d2-4429-97fa-6612872bb863
md"## Diffusion analysis"

# ╔═╡ d716423e-7945-4e0a-a6ab-17e0b94c721e
md"### Compounded metrics"

# ╔═╡ 0932e410-4a74-42b3-86c5-ac40f6be3543
md"### Dimensionality reduction and clustering"

# ╔═╡ 563c5195-fd67-48d8-9b01-a73ea756a7ba
md"Sampling:"

# ╔═╡ 228f2d0b-b964-4133-b0bc-fee7d9fe298f
begin
	sample_size = min(512, length(model.voters))
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)
end

# ╔═╡ 462930eb-f995-48f0-9564-0c3b3d3b437f
model_log = load_log(logger.exp_dir, 0)

# ╔═╡ 05eedffd-c82a-4dbd-8608-5fa00f9d0bae
sampled_voters = get_voters(model_log)[sampled_voter_ids]

# ╔═╡ 9f7f56db-8a73-40f7-bb47-9570e41a634f
md"Load voters from model:"

# ╔═╡ b203aed5-3231-4e19-a961-089c0a4cf8c6
md"Dimensionality reduction for visualisation of high dimensional opinions"

# ╔═╡ 868986e9-09f5-483e-9b8e-11b5ab6082fd
md"""Dimensionality reduction method: $(@bind dim_reduction_method Select(["PCA", "Tsne", "MDS"], default="PCA"))"""

# ╔═╡ 54855884-22b3-42e0-9120-c5f049043899
out_dim = 2

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

# ╔═╡ e80cfc91-fc51-4216-a74c-39038d77c9db
begin
	_model_log = load_log(logger.exp_dir, 0)
	_projections = reduce_dims(sampled_voters, dim_reduction_config)
	
	x_projections = _projections[1, 1:length(candidates)]
	y_projections = _projections[2, 1:length(candidates)]
	
	min_x, max_x = minimum(_projections[1, :]), maximum(_projections[1, :]) 
	min_y, max_y = minimum(_projections[2, :]), maximum(_projections[2, :])
end

# ╔═╡ 008cbbe1-949d-4be7-9178-9bfa27c6e2a8
begin
	projections = reduce_dims(sampled_voters, dim_reduction_config)
	#unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)
	projections
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

# ╔═╡ a652cf9b-adb5-47d2-906f-a7b479face45
OpinionDiffusion.model_vis2(model, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
labels, clusters = clustering(sampled_voters, clustering_config)

# ╔═╡ f0eed2be-799f-46bb-968b-b70462cc06fd
title = dim_reduction_method * "_" * clustering_method * "_" * string(length(sampled_voters))

# ╔═╡ 0c2345e5-c1b2-4af8-ad31-617ddc99257a
md"### Metrics specific for every timestamp"

# ╔═╡ a9a60036-1179-493b-b828-19422ecb9bdb
log_idxs = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(logger.exp_dir) if split(file, "_")[1] == "model"])

# ╔═╡ 83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
if step > 0
	prev_sampled_opinions = reduce(hcat, get_opinion(load_log(logger.exp_dir, step-1).voters[sampled_voter_ids]))
	difference = sampled_opinions - prev_sampled_opinions
	changes = vec(sum(abs.(difference), dims=1))
	println(sum(changes))
	draw_heat_vis(projections, changes, "Heat map")
end

# ╔═╡ f3dd8bf9-41d4-404a-8ab7-c3859be0de60
#visualizations = OpinionDiffusion.gather_vis(logger.exp_dir, sampled_voter_ids, dim_reduction_config, clustering_config, parties, candidates)

# ╔═╡ bb0c5066-169b-4d1a-a40c-dcbc2291b365
Plots.histogram2d(projections[1, :], projections[2, :], nbins=20, ylabel="Candidates", xlabel="Position")

# ╔═╡ 5f7dd641-8324-4b7b-9577-8188a22c8a8d
begin 
	social_network = get_social_network(model)
    voters = get_voters(model)
	
    cluster_graph = OpinionDiffusion.get_cluster_graph(model, clusters, labels, projections)
	
    println(Graphs.modularity(cluster_graph, labels))
end

# ╔═╡ abdc8a0c-6a15-4350-a3fd-3eeaec9daa10
Graphs.nv(cluster_graph)

# ╔═╡ 8c198f2d-2cef-4110-a9b3-7197ce20685c
cluster_metrics = OpinionDiffusion.cluster_graph_metrics(cluster_graph, social_network, voters, length(candidates))

# ╔═╡ e706a32f-d9c3-4f09-bd50-b8b43499f511
OpinionDiffusion.draw_cluster_graph(cluster_graph, cluster_metrics)

# ╔═╡ 0d3d101f-49bb-4a13-a7a9-1c9b1f97ddb5
visualizations = OpinionDiffusion.gather_vis2(logger.exp_dir, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ 4f2e171b-7914-4134-907e-222e8f8bbb68
@bind clk Clock()

# ╔═╡ ab3d3d53-4827-42ca-bc2d-abb1c4614c76
t = clk % length(visualizations) + 1

# ╔═╡ e618efd6-8819-4d5b-bb62-844c8466078b
visualizations[t]

# ╔═╡ 228b9f9c-1bde-4927-8c85-6b5938dcdbf1
collect(sort(frequencies, rev=true; byvalue=true))[1:min(length(frequencies), 10)]

# ╔═╡ 9b8fc5b4-9918-4c6b-8389-b19d5faf16fc
visualizations["distances"][t + 1]

# ╔═╡ 53b63024-6f60-4121-971d-50a6205c05a8
visualizations["voters"][t + 1]

# ╔═╡ c7386f6d-451c-49d7-9fd4-1277b734f7f0
t > 0 ? visualizations["heatmaps"][t] : nothing

# ╔═╡ a1681b3e-340f-4c8a-a991-98c8217a05ad
[candidate.party for candidate in candidates]

# ╔═╡ b771f5b9-42e1-444f-b5ac-cba2f0162e8d
[sortperm(borda_voting(get_votes(sampled_voters[collect(cluster)]), length(candidates), true), rev=true) for cluster in clusters if length(cluster) != 0]

# ╔═╡ 7bff8689-580e-43b5-9614-65b6f482eb5c
countss = [get_counts(get_votes(sampled_voters[collect(cluster)]), length(candidates)) for cluster in clusters if length(cluster) != 0]

# ╔═╡ 7e47b5c8-3ad1-4093-85e5-6b20238e9569
plots = Plots.plot([Plots.heatmap(count, yticks=1:length(src_candidates), xticks=1:length(src_candidates), ylabel="Candidate", xlabel="Position") for count in countss]..., layout = (length(candidates), 1), size = (669,900))

# ╔═╡ e814359f-3e70-48ab-9b57-aed8bcaf0102
clusters

# ╔═╡ cd8ffe9d-e990-43ed-ad77-28067d1536ed
draw_degree_distr(Graphs.degree_histogram(model_log.social_network))

# ╔═╡ 10704554-3d11-428e-b05f-2747ae86af92
md"### Compare Two runs"

# ╔═╡ 7f6914c0-4b6a-4923-8069-637afade93e1


# ╔═╡ 59039b0b-6225-4fb6-a43b-eb0eadcf5e4c


# ╔═╡ a0cd5f7e-1267-4e2c-bb71-873d5096a7da
md"### Node visualization"

# ╔═╡ 53586e20-27c6-4804-8d22-4b78ca6adf74
node_id = 42

# ╔═╡ bcf995a9-7f73-444c-adbc-fb13571112e9
x, y = [-1.0,2.2,3.1], [4.1,-4.2,4.2]#ego_project[1, :], ego_project[2, :]

# ╔═╡ 5a792e2e-909b-4690-a0bf-d05afe7f4c81
function ego(social_network, node_id, depth)
    neighs = Graphs.neighbors(social_network, node_id)
	ego_nodes = Set(neighs)
    push!(ego_nodes, node_id)
	
    front = Set(neighs)
    for i in 1:depth - 1
        new_front = Set()
        for voter in front
            union!(new_front, Graphs.neighbors(social_network, voter))
        end
        front = setdiff(new_front, ego_nodes)
        union!(ego_nodes, front)
    end

    return Graphs.induced_subgraph(social_network, collect(ego_nodes))
end

# ╔═╡ 596452b2-e74b-4a20-b3e0-3a918ffef042
egon = ego(model.social_network, 100, 1)

# ╔═╡ 84d258fb-bd87-4d37-a3d1-bc8f43477da3
typeof(egon[1])

# ╔═╡ 18c07cb1-3a91-405c-a626-f608e0d1ea9d
OpinionDiffusion.GraphPlot.gplot(egon[1], x, y, nodesize=[Graphs.degree(model.social_network, node) for node in egon[2]]) #nodefillc=cluster_colors[labels[egon[2]]])

# ╔═╡ ceadcd2d-42ac-4332-a49c-589cde3d500d
depth = 1.0

# ╔═╡ bb65c59a-5f29-4453-a5a5-dae5856347c0
length(Graphs.neighborhood(model.social_network, node_id, depth))

# ╔═╡ 52dc63c8-c864-4075-91d8-8c0a98a6a717
Graphs.induced_subgraph(model.social_network, Graphs.neighborhood(model.social_network, node_id, depth))

# ╔═╡ 141d26c0-65f2-4266-b5a8-5b4b4c28f16e
#ego_project = reduce_dim(sampled_opinions, reduce_dim_config)[:, egon[2]]

# ╔═╡ 87320bc9-e825-4aa8-84ea-9fd75b7ff4fd
md"#### Compare ensemble diffusions"

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

# ╔═╡ 48a2a821-f104-49da-b4e3-de7280f2eb03
model_configs = [OpinionDiffusion.load(log, "model_config") for log in ensemble_logs]

# ╔═╡ e8971acb-dc93-4a01-85f8-5190f1972ece
typeof(model_configs[1].voter_init_config)

# ╔═╡ 5c505323-4487-4433-8ea7-4357b0fb8166
[OpinionDiffusion.load(log, "diffusion_config") for log in ensemble_logs]

# ╔═╡ a68fa858-2751-451d-a8f3-5c34c95e8b04
 #ensemble_logs = [logger.model_dir * "/" * file for file in readdir(logger.model_dir) if split(file, "_")[1] == "ensemble"][2:end]

# ╔═╡ 4c140c0e-71c7-4567-b37e-286395a450a3
md"
#### What statistics do we care about during diffusion:
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
#### What visualizations are important to understand specific state of society:
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

# ╔═╡ 93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
function init_metrics(model, can_count)
    g = get_social_network(model)
    voters = get_voters(model)

	histogram = Graphs.degree_histogram(g)
    keyss = collect(keys(histogram))
	
	metrics = Dict()
	metrics["min_degrees"] = [minimum(keyss)]
	metrics["avg_degrees"] = [Graphs.ne(g) * 2 / Graphs.nv(g)]
    metrics["max_degrees"] = [maximum(keyss)]
	metrics["avg_edge_dist"] = [OpinionDiffusion.StatsBase.mean(OpinionDiffusion.get_edge_distances(g, voters))]
    metrics["clustering_coeff"] = [Graphs.global_clustering_coefficient(g)]
	#metrics["diameter"] = [Graphs.diameter(g)]
    
	#election results
	votes = get_votes(voters)
	metrics["avg_vote_length"] = [OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes])]
    metrics["unique_votes"] = [length(unique(votes))]
	
    metrics["plurality_votings"] = [plurality_voting(votes, can_count, true)]
    metrics["borda_votings"] = [borda_voting(votes, can_count, true)]
    #metrics["copeland_votings"] = [copeland_voting(votes, can_count)]

	metrics["positions"] = [get_positions(voters, can_count)]
	
	return metrics
end

# ╔═╡ 2d564bf0-2584-4da3-9890-40b56b023915
metrics = init_metrics(model, length(candidates))

# ╔═╡ c13d3ec3-95f1-4e85-840c-4cb1dad8eaca
metrics

# ╔═╡ e31cdd5b-3310-43fb-addc-96144547de2b
update_metrics!(model, metrics) = update_metrics!(model, metrics, length(candidates))

# ╔═╡ 8d259bcc-d54c-401f-b864-87701f2bcf46
function update_metrics!(model, diffusion_metrics, can_count)
    g = get_social_network(model)
    voters = get_voters(model)

    dict = Graphs.degree_histogram(g)
    keyss = collect(keys(dict))
	
    push!(diffusion_metrics["min_degrees"], minimum(keyss))
    push!(diffusion_metrics["avg_degrees"], Graphs.ne(g) * 2 / Graphs.nv(g))
    push!(diffusion_metrics["max_degrees"], maximum(keyss))
	push!(diffusion_metrics["avg_edge_dist"], OpinionDiffusion.StatsBase.mean(OpinionDiffusion.get_edge_distances(g, voters)))
    push!(diffusion_metrics["clustering_coeff"], Graphs.global_clustering_coefficient(g))
    #push!(diffusion_metrics["diameter"], Graphs.diameter(g))
    
    votes = get_votes(voters)
	push!(diffusion_metrics["avg_vote_length"], OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes]))
    push!(diffusion_metrics["unique_votes"], length(unique(votes)))
	
    push!(diffusion_metrics["plurality_votings"], plurality_voting(votes, can_count, true))
    push!(diffusion_metrics["borda_votings"], borda_voting(votes, can_count, true))
    #push!(diffusion_metrics["copeland_votings"], copeland_voting(votes, can_count))
    push!(diffusion_metrics["positions"], get_positions(voters, can_count))
end

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_run
	if ensemble_size == 1
		for i in 1:diffusions
			run!(model, diffusion_config, logger)
			update_metrics!(model, metrics, length(candidates))
		end
	elseif !ensemble_model
		result = OpinionDiffusion.run_ensemble(model, ensemble_size, diffusions, metrics, update_metrics!, diffusion_config, logger)
		gathered_metrics = gather_metrics([diffusion["metrics"] for diffusion in result])
	else
		result = OpinionDiffusion.run_ensemble_model(ensemble_size, diffusions, election, init_metrics, length(candidates), update_metrics!, model_config, diffusion_config, true)
		gathered_metrics = gather_metrics([diffusion["metrics"] for diffusion in result])
	end
end

# ╔═╡ 12ffda44-7cee-48a7-999b-2224c3549dea
gathered_metrics

# ╔═╡ 6deeecf9-f727-4b5d-82c5-f4f9a85eb55c
result

# ╔═╡ a4f875d7-685e-43bb-a806-be6ae6547ffb
extremes = extreme_runs(result, "plurality_votings", 4)

# ╔═╡ 3eb2368f-5cdf-4f43-84fa-478865936371
min_model_seed, min_diffusion_seed = result[extremes[1]]["model_seed"], result[extremes[1]]["diffusion_seed"]

# ╔═╡ 5cd2dbf8-5a55-4690-b16d-8b1432015054
max_model_seed, max_diffusion_seed = result[extremes[2]]["model_seed"], result[extremes[2]]["diffusion_seed"]

# ╔═╡ c86a8e9f-c5e9-4931-8923-dcf114df3118
if anal_run
	min_logger = OpinionDiffusion.run(election, length(candidates), model_config, min_model_seed, diffusion_config, diffusions, min_diffusion_seed)

	max_logger = OpinionDiffusion.run(election, length(candidates), model_config, max_model_seed, diffusion_config, diffusions, max_diffusion_seed)
end

# ╔═╡ 9883316d-c845-4a4d-a4f8-b8022727cb0b
min_log_idxs = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(min_logger.exp_dir) if split(file, "_")[1] == "model"])

# ╔═╡ e518a2db-a02e-4cfb-8587-f892fd9cbc85
gathered_metrics

# ╔═╡ 66267f11-eac9-4fbd-814f-89897f74a046
function ensemble_init_model_AB(ensemble_size, election, can_count, init_metrics, update_metrics, model_config, diffusion_config)
	popularity_metrics = []
	
	sizes = collect(100:100:length(election))
	for popularity_ratio in collect(0.0:0.5:1.0)
		model_config = General_model_config(
							voter_init_config = voter_type,
							graph_init_config = weighted_AB_graph_config(
							m=5, 
							popularity_ratio=popularity_ratio),
							)
		metrics = []
		for size in sizes
			gathered_metrics = OpinionDiffusion.run_ensemble_model(ensemble_size, 0, election[1:size], init_metrics, can_count, update_metrics!, model_config, diffusion_config, false)
			gathered_metrics["size"] = [size]
			push!(metrics, gathered_metrics)
		end

		push!(popularity_metrics, metrics)
	end

	return popularity_metrics
end

# ╔═╡ de7a5c20-d5d6-4fe5-b7c3-561b17a90143
function ensemble_init_model_(ensemble_size, election, can_count, init_metrics, update_metrics, model_configs, diffusion_config)
	size_metrics = []
	
	sizes = collect(100:100:length(election))
	for size in sizes
		metrics = []

		for model_config in model_configs
			gathered_metrics = OpinionDiffusion.run_ensemble_model(ensemble_size, 0, election[1:size], init_metrics, can_count, update_metrics!, model_config, diffusion_config, false)
			
			gathered_metrics["size"] = [size]
			push!(metrics, gathered_metrics)
		end

		push!(size_metrics, metrics)
	end

	metrics = [deepcopy(size_metrics[1][1]), deepcopy(size_metrics[1][2]), deepcopy(size_metrics[1][3])]
	for i in 2:length(sizes)
		for j in 1:length(pops)
			for (metric, val) in size_metrics[i][j]
				push!(metrics[j][metric], val[1])
			end
		end
	end
	
	return metrics
end

# ╔═╡ eccddfc6-2e57-4f9e-88f3-88166fc5da11
function ensemble_init_model(BA, ensemble_size, election, can_count, init_metrics, update_metrics, diffusion_config)
	size_metrics = []
	
	for size in collect(100:100:length(election))
		homophily_metrics = []
		
		for homophily_ratio in collect(0.0:0.5:1.0)
			model_config = BA ? 
			General_model_config(
							voter_init_config = voter_type,
							graph_init_config = BA_graph_config(
							m=5, 
							homophily=homophily_ratio),
			) : General_model_config(
							voter_init_config = voter_type,
							graph_init_config = DEG_graph_config(
								exp = 1.0,
								scale = 2.0,
								max_degree=min(500, size / 4),
								target_cc = 0.5, 
								homophily = homophily_ratio
								)
							)
			
			gathered_metrics = OpinionDiffusion.run_ensemble_model(ensemble_size, 0, election[1:size], init_metrics, can_count, update_metrics!, model_config, diffusion_config, false)
			
			gathered_metrics["size"] = [size]
			push!(homophily_metrics, gathered_metrics)
		end

		push!(size_metrics, homophily_metrics)
	end

	metrics = [deepcopy(size_metrics[1][1]), deepcopy(size_metrics[1][2]), deepcopy(size_metrics[1][3])]
	for i in 2:length(sizes)
		for j in 1:length(pops)
			for (metric, val) in size_metrics[i][j]
				push!(metrics[j][metric], val[1])
			end
		end
	end
	
	return metrics
end

# ╔═╡ 187da043-ee48-420a-91c7-5e3a4fdc30bb
function draw_voting_rules(data, voting_rules, candidates, parties)
	n = length(voting_rules)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)

	for (i, metric) in enumerate(voting_rules)
		result = transpose(reduce(hcat, data[metric]))
		OpinionDiffusion.draw_voting_res!(plot[i, 1], candidates, parties, result, metric)
	end
	
	return Plots.plot(plot)
end

# ╔═╡ 840c2562-c444-4422-9cf8-e82429163627
draw_voting_rules(gathered_metrics, ["plurality_votings", "borda_votings", "positions"], candidates, parties)

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
# ╠═75588cfd-52c9-4406-975c-c03158db6e78
# ╟─55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
# ╠═ff873978-d93f-4ba2-aadd-6cfd3b136e3d
# ╟─46e8650e-f57d-48d7-89de-1c72e12dea45
# ╟─bcc5468c-2a49-409d-b810-05fc30f4edca
# ╠═c6ccf2a8-e045-4da9-bbdb-270327c2d53f
# ╟─c8522415-1fd9-4c06-bf2a-38ab23153b56
# ╠═21f341c3-4d38-4449-b0f8-62e4ab14c99b
# ╠═042ec90d-8c0b-4acb-8d4c-31f701876cb6
# ╠═06d301e2-90c1-4fdf-b5ab-e127f8dee777
# ╠═d5d394c0-9b7c-4255-95fd-dc8cc32ca018
# ╟─20078431-5a7c-4a6f-b181-984ed54cf506
# ╟─8c709e84-66f8-4128-a85c-66a4c5ffc9b7
# ╟─977d39e2-7f82-49e8-a93f-889204bd19cb
# ╠═ad94415f-fc56-4a2f-9068-e05d8633aafe
# ╠═93b8909b-479c-422a-b475-2befadf5e9ec
# ╠═8ea22c93-1fe3-44b2-88c1-fb6ccd195866
# ╠═28f103a9-2b18-4cfc-bcf3-34d512f8da03
# ╟─20893df2-fa42-46bf-889e-582b9ac39164
# ╟─831816fa-7983-4c05-9a76-a03631bf1e57
# ╟─40fdc182-c693-48fb-99ee-43d1bc78d95f
# ╠═a38a2e2e-e742-4c0b-9cf5-cd8376178300
# ╠═342d3d34-4a3a-4f2d-811b-e9ab143504fd
# ╠═c2d75170-789d-41b4-b30b-40b143d23702
# ╠═825b45d1-fe6a-4460-8fe2-b323019c56b6
# ╟─ed315e83-6d73-4f9a-afb9-f0174e08ef29
# ╟─a87ea61d-13d9-4c91-b08e-9f24cde3d290
# ╠═10cb247f-d445-4091-b863-49deeb4c35fe
# ╟─c91395e7-393a-4ee1-ab44-7269cb1314d8
# ╟─48a2cf2b-bd86-4131-a230-290124cc5f48
# ╠═3b67247a-7dd8-49e7-ad88-64a8dad90d17
# ╠═25374289-3cf1-4d27-887f-a47f9794f2bc
# ╠═7069907d-2949-4f68-9602-9bef9ff46064
# ╟─c4dfe306-aad8-4248-bc9e-c2de841a7354
# ╠═738b7617-00c5-4d25-ae1b-61788ba23f5c
# ╟─39256cbd-7807-42bc-81b1-d6f2128ccaf9
# ╠═7e0d083d-2de1-4a4c-8d19-7dea4f95152a
# ╟─338aa155-a4fb-4ff9-b27d-fc5481abb58d
# ╠═0d3b58fe-8205-4d4c-9174-09cec37a61f3
# ╟─759557c0-228f-4827-9038-1963b54a08d9
# ╠═776f99ea-6e44-4eb6-b2dd-3552bbb39954
# ╠═b4d60582-ab23-4b2e-84bb-efd60786dc93
# ╠═637a053e-1ce8-4107-926a-46f46b4718f8
# ╠═985131a7-7c11-4f9d-ae00-ef031002592d
# ╠═37cf8471-02a7-4a91-b17e-f44d06974d97
# ╠═d8b613c5-7276-466d-a54a-7670f0921b35
# ╠═a35b87d8-76eb-4b3e-b195-19dd5564bcf8
# ╟─4712586c-bc93-43df-ae66-1e75a21b6f85
# ╠═571e7a33-20b7-4432-b553-c07b9081c68d
# ╟─3cede6ac-5765-4c72-9b53-103c9c6a9bd9
# ╟─98885ec6-7561-43d7-bdf6-7f58fb2720f6
# ╟─72450aaf-c6c4-458e-9555-39c31345116b
# ╟─8e0750a9-2b83-4652-805b-dc1be2484161
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╠═fa47c89c-784f-4552-9e34-22e499a0231f
# ╠═93cb669f-6743-4b50-80de-c8594ea20497
# ╟─2c837f03-3329-4ab3-ad31-b0fe79df6bb7
# ╠═2d564bf0-2584-4da3-9890-40b56b023915
# ╠═c13d3ec3-95f1-4e85-840c-4cb1dad8eaca
# ╠═a652cf9b-adb5-47d2-906f-a7b479face45
# ╠═29cf96fa-4b08-4909-a70e-288441da15ad
# ╠═9609ec4b-ed07-4121-8db3-f7d374698ac0
# ╠═923cde07-7b5a-46a8-86a8-6c076c61631b
# ╠═3bea0fad-f106-49f4-be64-2ba7361591a6
# ╠═6f5eb499-b01d-4e05-a380-04ac17345b68
# ╠═05ad5115-2a2b-479e-96d1-9796572d2f54
# ╠═07893eca-c6ea-411a-94f5-802134c46311
# ╠═868709ae-5084-42db-a3c4-94fe2180bb05
# ╠═045b7c19-a26d-4237-8a50-55abf7f9b57c
# ╠═bb65c59a-5f29-4453-a5a5-dae5856347c0
# ╠═52dc63c8-c864-4075-91d8-8c0a98a6a717
# ╟─09540198-bead-4df1-99bd-6e7848e7d215
# ╟─6f40b5c4-1252-472c-8932-11a2ee0935d2
# ╠═b6cd7a31-80b9-49eb-8004-34de5e6ad910
# ╠═0295587e-ad65-4bf1-b6d4-1b87a1e844ff
# ╠═63a4233d-c596-4844-8539-91a2223f2266
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╠═d877c5d0-89af-48b9-bcd0-c1602d58339f
# ╠═27a60724-5d19-419f-b208-ffa0c78e2505
# ╠═87c573c1-69a4-4a61-bbb8-acb716f8ec6d
# ╟─de772425-25de-4228-b12e-d567b8ceb20f
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═12ffda44-7cee-48a7-999b-2224c3549dea
# ╠═f2ca9891-3663-4ec6-bc69-e454faabba53
# ╠═0009fe9c-117f-433c-b65f-2a0835aca0b9
# ╠═6deeecf9-f727-4b5d-82c5-f4f9a85eb55c
# ╠═a4f875d7-685e-43bb-a806-be6ae6547ffb
# ╠═3eb2368f-5cdf-4f43-84fa-478865936371
# ╠═5cd2dbf8-5a55-4690-b16d-8b1432015054
# ╟─3fe9215c-ba2f-4aaa-bcb0-1eb8f1982db8
# ╠═c86a8e9f-c5e9-4931-8923-dcf114df3118
# ╠═1173f5f8-b355-468c-8bfb-beebff5ba12b
# ╠═0cf5fa96-ff32-4853-b6c7-ef1843f5281f
# ╠═66267f11-eac9-4fbd-814f-89897f74a046
# ╠═de7a5c20-d5d6-4fe5-b7c3-561b17a90143
# ╠═eccddfc6-2e57-4f9e-88f3-88166fc5da11
# ╠═56d7f1d9-f291-446b-8281-013d017102f2
# ╠═ccbfad20-549e-4275-8bbb-644157d98926
# ╠═409b3357-dcf0-4ddb-8a4b-b7e2f21cee8f
# ╠═19c42165-d4ba-4c86-b134-514f6b017fe9
# ╠═13624365-3fba-49c7-bbe0-b33f9865e953
# ╠═f560c9f3-1b7d-48c0-b8d7-40f4b43d3cec
# ╠═8fd45292-6704-4b57-b7e4-b650dce8d19c
# ╠═a1714045-adf8-427e-a960-dad1fda7aaa3
# ╠═2d3e4761-d868-42ee-99fa-40f2a9ee522b
# ╟─aad0da61-59d2-4429-97fa-6612872bb863
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═e518a2db-a02e-4cfb-8587-f892fd9cbc85
# ╠═09d34d24-0fb0-4cc6-8ab6-c0d55b3346d0
# ╠═840c2562-c444-4422-9cf8-e82429163627
# ╟─0932e410-4a74-42b3-86c5-ac40f6be3543
# ╟─563c5195-fd67-48d8-9b01-a73ea756a7ba
# ╠═228f2d0b-b964-4133-b0bc-fee7d9fe298f
# ╠═462930eb-f995-48f0-9564-0c3b3d3b437f
# ╠═05eedffd-c82a-4dbd-8608-5fa00f9d0bae
# ╟─9f7f56db-8a73-40f7-bb47-9570e41a634f
# ╟─b203aed5-3231-4e19-a961-089c0a4cf8c6
# ╠═868986e9-09f5-483e-9b8e-11b5ab6082fd
# ╠═54855884-22b3-42e0-9120-c5f049043899
# ╠═390be9ec-29a1-4138-952e-fc4eb5eb2ecb
# ╠═e80cfc91-fc51-4216-a74c-39038d77c9db
# ╠═008cbbe1-949d-4be7-9178-9bfa27c6e2a8
# ╟─52893c8c-d1b5-482a-aae7-b3ec5c590b77
# ╠═e60db281-0bd0-4eb6-94fa-4c30766464fd
# ╠═ca8a24e4-ce3b-47d9-ad39-8507fa910a9d
# ╠═c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
# ╠═f0eed2be-799f-46bb-968b-b70462cc06fd
# ╟─0c2345e5-c1b2-4af8-ad31-617ddc99257a
# ╠═a9a60036-1179-493b-b828-19422ecb9bdb
# ╠═83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
# ╠═7e47b5c8-3ad1-4093-85e5-6b20238e9569
# ╠═f3dd8bf9-41d4-404a-8ab7-c3859be0de60
# ╠═bb0c5066-169b-4d1a-a40c-dcbc2291b365
# ╠═5f7dd641-8324-4b7b-9577-8188a22c8a8d
# ╠═abdc8a0c-6a15-4350-a3fd-3eeaec9daa10
# ╠═8c198f2d-2cef-4110-a9b3-7197ce20685c
# ╠═e706a32f-d9c3-4f09-bd50-b8b43499f511
# ╠═0d3d101f-49bb-4a13-a7a9-1c9b1f97ddb5
# ╠═4f2e171b-7914-4134-907e-222e8f8bbb68
# ╠═ab3d3d53-4827-42ca-bc2d-abb1c4614c76
# ╠═e618efd6-8819-4d5b-bb62-844c8466078b
# ╠═228b9f9c-1bde-4927-8c85-6b5938dcdbf1
# ╠═9b8fc5b4-9918-4c6b-8389-b19d5faf16fc
# ╠═53b63024-6f60-4121-971d-50a6205c05a8
# ╠═c7386f6d-451c-49d7-9fd4-1277b734f7f0
# ╠═a1681b3e-340f-4c8a-a991-98c8217a05ad
# ╠═b771f5b9-42e1-444f-b5ac-cba2f0162e8d
# ╠═7bff8689-580e-43b5-9614-65b6f482eb5c
# ╠═e814359f-3e70-48ab-9b57-aed8bcaf0102
# ╠═cd8ffe9d-e990-43ed-ad77-28067d1536ed
# ╟─10704554-3d11-428e-b05f-2747ae86af92
# ╠═9883316d-c845-4a4d-a4f8-b8022727cb0b
# ╠═7f6914c0-4b6a-4923-8069-637afade93e1
# ╠═59039b0b-6225-4fb6-a43b-eb0eadcf5e4c
# ╠═a0cd5f7e-1267-4e2c-bb71-873d5096a7da
# ╠═53586e20-27c6-4804-8d22-4b78ca6adf74
# ╠═596452b2-e74b-4a20-b3e0-3a918ffef042
# ╠═84d258fb-bd87-4d37-a3d1-bc8f43477da3
# ╠═bcf995a9-7f73-444c-adbc-fb13571112e9
# ╠═18c07cb1-3a91-405c-a626-f608e0d1ea9d
# ╠═5a792e2e-909b-4690-a0bf-d05afe7f4c81
# ╠═ceadcd2d-42ac-4332-a49c-589cde3d500d
# ╠═141d26c0-65f2-4266-b5a8-5b4b4c28f16e
# ╟─87320bc9-e825-4aa8-84ea-9fd75b7ff4fd
# ╠═af643514-f20f-436b-966d-05021b566b1e
# ╠═4add7dc5-0af8-4179-a496-a46767cc85ef
# ╠═fa876197-2401-4b54-8b56-e4abdf8f8801
# ╠═48a2a821-f104-49da-b4e3-de7280f2eb03
# ╠═e8971acb-dc93-4a01-85f8-5190f1972ece
# ╠═5c505323-4487-4433-8ea7-4357b0fb8166
# ╠═a68fa858-2751-451d-a8f3-5c34c95e8b04
# ╠═f96c982d-b5db-47f7-91e0-9c9b3b36332f
# ╠═fc90baf0-a51d-4f3f-b036-b3bc381b2dbc
# ╟─4c140c0e-71c7-4567-b37e-286395a450a3
# ╟─f539cf71-34ae-4e22-a7ac-d259b55cb2d3
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═e31cdd5b-3310-43fb-addc-96144547de2b
# ╠═8d259bcc-d54c-401f-b864-87701f2bcf46
# ╠═187da043-ee48-420a-91c7-5e3a4fdc30bb
# ╠═ff89eead-5bf5-4d52-9134-aa415ae156b7
# ╠═e70e36ad-066e-4619-a06a-56e325745a0e
# ╠═b94c9f37-b321-4db2-9da9-75917be8e52e
