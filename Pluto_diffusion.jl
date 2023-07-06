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
md"## Dataset"

# ╔═╡ bcc5468c-2a49-409d-b810-05fc30f4edca
md"""Select dataset: $(@bind input_filename Select([filename for filename in readdir("./data") if split(filename, ".")[end] == "toc"], default = "ED-00001-00000002.toc"))"""

# ╔═╡ 2f08ff39-d1d6-4341-8da8-5c46e23ae856


# ╔═╡ c6ccf2a8-e045-4da9-bbdb-270327c2d53f
init_election = parse_data("./data/" * input_filename)

# ╔═╡ c8522415-1fd9-4c06-bf2a-38ab23153b56
md"### Dataset Summary"

# ╔═╡ 61fa4389-3b1e-4be3-a5a2-ee37218d7d55
draw_election_summary(get_election_summary(get_votes(init_election), drop_last=true))

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

# ╔═╡ 20893df2-fa42-46bf-889e-582b9ac39164
md"## Configure model"

# ╔═╡ 831816fa-7983-4c05-9a76-a03631bf1e57
md"### Voter config"

# ╔═╡ 40fdc182-c693-48fb-99ee-43d1bc78d95f
md"#### Weighting of Spearman voter"

# ╔═╡ 342d3d34-4a3a-4f2d-811b-e9ab143504fd
begin
	weight_func = position -> (length(candidates) - position)^weighting_rate
	weights = Vector{Float64}(undef, length(candidates))
	weights[1] = 0.0
	for i in 2:length(weights)
		weights[i] = weights[i - 1] + weight_func(i - 1) + 1
	end
	max_sp_distance = OpinionDiffusion.get_max_distance(length(candidates), weights)
	weights = weights ./ max_sp_distance
	weights
end

# ╔═╡ 825b45d1-fe6a-4460-8fe2-b323019c56b6
Plots.bar([string(i) * '-' * string(i + 1) for i in 1:length(candidates)-1], [weight_func(x) for x in 1:length(candidates)-1], legend=false, title="Bucket Distances")

# ╔═╡ ed315e83-6d73-4f9a-afb9-f0174e08ef29
md"#### Voter selection"

# ╔═╡ a87ea61d-13d9-4c91-b08e-9f24cde3d290
md"""Select voter type: $(@bind voter_type Select(["Kendall-tau voter", "Spearman voter"], default="Spearman voter"))"""

# ╔═╡ 10cb247f-d445-4091-b863-49deeb4c35fe
if voter_type == "Spearman voter"
		voter_config = Spearman_voter_config(
		weighting_rate = 0.0
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

# ╔═╡ 341a601f-2727-4362-b7d1-29dd47a92539
md"### Model config"

# ╔═╡ 759557c0-228f-4827-9038-1963b54a08d9
md"### Locate model"

# ╔═╡ 637a053e-1ce8-4107-926a-46f46b4718f8
log_dir_path = "./logs/"

# ╔═╡ 345c5270-0822-4ba6-b326-14a60d8a8d46
md"""Select model: $(@bind model_dir Select([filename for filename in readdir( log_dir_path) if split(filename, "_")[1] == "model" && split(filename, "_")[2] != "ensemble"]))"""

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir_path = log_dir_path * model_dir

# ╔═╡ 3cede6ac-5765-4c72-9b53-103c9c6a9bd9
md"## Create/load model"

# ╔═╡ 98885ec6-7561-43d7-bdf6-7f58fb2720f6
md"Choose source of the model and then check execution barrier for generation of the model"

# ╔═╡ 72450aaf-c6c4-458e-9555-39c31345116b
md"""
Model source: $(@bind model_source Select(["load_model" => "Load", "new_model" => "New (takes time)"]))
"""

# ╔═╡ 8e0750a9-2b83-4652-805b-dc1be2484161
md"""
Logging $(@bind logging CheckBox())
"""

# ╔═╡ 1937642e-7f63-4ffa-b01f-22208a716dac
md"""
---
**Execution barrier** $(@bind cb_model CheckBox())

---
"""

# ╔═╡ 9b6e56ed-cd89-4684-83d7-b2f931a07d14
General_model_config(
	voter_config=voter_config,
	graph_config=graph_config
)

# ╔═╡ 3671ba34-71bc-4e11-ad5e-aaeecf3f1e3f
if logging
	logger = Model_logger(model, )
end

# ╔═╡ 2c837f03-3329-4ab3-ad31-b0fe79df6bb7
md"## Initial model metrics"

# ╔═╡ a13fff49-bc77-492c-8402-3fec14e9a7e4


# ╔═╡ a6590191-0fbd-41a9-a7c3-b99e68bb27aa
begin
	sample_size = length(model.voters)
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)
end

# ╔═╡ 988a9fcb-ad5f-4be6-b5f5-c3b05587a8d4
xs, mean_y = aggregate_mean(degrees, ccs)

# ╔═╡ 4f7384d8-4c16-4a75-93ca-755215d74643
begin
	f = Figure()
	OpinionDiffusion.draw_degree_cc!(f[1,1], get_social_network(model))
	f
end

# ╔═╡ 09540198-bead-4df1-99bd-6e7848e7d215
md"## Configure diffusion"

# ╔═╡ 6f40b5c4-1252-472c-8932-11a2ee0935d2
md"Setup diffusion parameters and then check execution barrier for confirmation."

# ╔═╡ b6cd7a31-80b9-49eb-8004-34de5e6ad910
attract_proba = 1.0

# ╔═╡ e3f2c391-4fc3-47b5-bad2-5abc8f89a345
if voter_type == "Spearman voter"
	init_voter_diffusion = [SP_init_diff_config(init_sample_size, Distributions.Normal(0.5, 0.1)) for _ in 1:ensemble_size]
	
	voter_diffusion = SP_diff_config(
		rng=Random.MersenneTwister(rand(UInt32)),
		evolve_vertices=1.0,
		attract_proba = attract_proba,
		change_rate = 0.05,
		normalize_shifts = (true, weights[1], weights[end])
	)
else #Kendall voter
	init_voter_diffusion = KT_init_diff_config(init_sample_size, Distributions.Normal(0.5, 0.1))
	init_voter_diffusions = [KT_init_diff_config(init_sample_size, Distributions.Normal(0.5, 0.1)) for _ in 1:ensemble_size]
	
	voter_diffusion= KT_diff_config(
		rng=Random.MersenneTwister(rand(UInt32)),
		evolve_vertices=1.0,
		attract_proba = attract_proba
	)
end

# ╔═╡ e7f02f43-f8d5-4b7d-b097-4cbe7c7541a7
graph_diffusion = Graph_diff_config(
	rng=Random.MersenneTwister(rand(UInt32)),
	evolve_edges=0.0,
	homophily = homophily
)

# ╔═╡ 970cf900-f459-4eee-a0d3-226a40b6422f
init_diff_configs = [init_voter_diffusion]

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diff_configs = [voter_diffusion, graph_diffusion]

# ╔═╡ 971693bb-1a08-4266-a93b-c3e9d60d8bcd
md"""
Restart: $(@bind restart CheckBox())
"""

# ╔═╡ de772425-25de-4228-b12e-d567b8ceb20f
md"## Run diffusion"

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
---
**Execution barrier** $(@bind cb_run CheckBox())

---
"""

# ╔═╡ aad0da61-59d2-4429-97fa-6612872bb863
md"## Diffusion analysis"

# ╔═╡ d716423e-7945-4e0a-a6ab-17e0b94c721e
md"### Compounded metrics"

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

# ╔═╡ a652cf9b-adb5-47d2-906f-a7b479face45
vs = timestamp_vis(model, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ 5bf9306d-ed0f-4d12-af1f-75ba9fe1e9a4
vs[1][2]

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

# ╔═╡ e8971acb-dc93-4a01-85f8-5190f1972ece
typeof(model_configs[1].voter_init_config)

# ╔═╡ 5c505323-4487-4433-8ea7-4357b0fb8166
[OpinionDiffusion.load(log, "diffusion_config") for log in ensemble_logs]

# ╔═╡ a68fa858-2751-451d-a8f3-5c34c95e8b04
 #ensemble_logs = [logger.model_dir * "/" * file for file in readdir(logger.model_dir) if split(file, "_")[1] == "ensemble"][2:end]

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

# ╔═╡ d9fd7b27-7bd1-4e4d-a1b5-f4f173a86c2d
sizes = collect(100:100:length(election))

# ╔═╡ 545103ec-9f9b-4b24-a993-818a20f18f49


# ╔═╡ dba61064-49ec-4ed7-b8f0-429751318282
function variable_model(ensemble_size, election, can_count, init_metrics, update_metrics, model_configs, init_diffusion_configs, diffusion_configs)

end

# ╔═╡ 9b494231-4d23-4573-b686-e83119bafe88
function variable_init_diffusion(ensemble_size, model, init_metrics, update_metrics, init_diffusion_configs, diffusion_configs)

end

# ╔═╡ e35b5055-6ebd-49cd-83be-429f0f7dcaf5
function variable_diffusion(ensemble_size, model, init_metrics, update_metrics, diffusion_configs)
	
end

# ╔═╡ 93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
function init_metrics(model)
    g = get_social_network(model)
    voters = get_voters(model)
	candidates = get_candidates(model)
	can_count = length(candidates)
	
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

# ╔═╡ 8d259bcc-d54c-401f-b864-87701f2bcf46
function update_metrics!(model, diffusion_metrics)
    g = get_social_network(model)
    voters = get_voters(model)
	candidates = get_candidates(model)
	can_count = length(candidates)
	
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
		init_diffusion!(model, init_diff_configs)
		actions = run!(model, diff_configs, diffusions, logger=logger, checkpoint=checkpoint, metrics=metrics, update_metrics! =update_metrics!)

	elseif !ensemble_model
		result = run_ensemble(model, ensemble_size, diffusions, metrics, update_metrics!, diff_configs, logger)
		gathered_metrics = gather_metrics([diffusion["metrics"] for diffusion in result])
		
	else
		result = run_ensemble(
		    ensemble_size,
			ensemble_mode,
		    diffusions,
		    election,
		    candidates,
		    init_metrics,
		    update_metrics!,
		    model_configs,
		    init_diff_configs,
		    diff_configs,
		    true
		)
		result = run_ensemble_model(ensemble_size, diffusions, election, candidates, init_metrics, update_metrics!, model_config, init_diff_configs, diff_configs, true)
		gathered_metrics = gather_metrics([diffusion["metrics"] for diffusion in result])
	end
end

# ╔═╡ e518a2db-a02e-4cfb-8587-f892fd9cbc85
gathered_metrics

# ╔═╡ 7f898ae7-8613-43a6-95e1-f61748cec34a
function variable_size(sizes, ensemble_size, election, can_count, init_metrics, update_metrics, model_configs, diffusion_configs)
	size_metrics = []
	
	for size in sizes
		metrics = []

		for model_config in model_configs
			result = run_ensemble_model(ensemble_size, diffusions, election, candidates, init_metrics, update_metrics!, model_config, init_diff_configs, diff_configs, true)
			
			gathered_metrics = gather_metrics([diffusion["metrics"] for diffusion in result])
			
			gathered_metrics["size"] = [size]
			push!(metrics, gathered_metrics)
		end

		push!(size_metrics, metrics)
	end

	metrics = [deepcopy(size_metrics[1][i]) for i in 1:ensemble_size]
	for i in 2:length(sizes)
		for j in 1:length(pops)
			for (metric, val) in size_metrics[i][j]
				push!(metrics[j][metric], val[1])
			end
		end
	end

	return metrics
end

# ╔═╡ 1efa4e48-4a08-4aca-a255-8acd559c4bc8
variable_size(sizes)

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

# ╔═╡ 48a2a821-f104-49da-b4e3-de7280f2eb03
model_configs = [OpinionDiffusion.load(log, "model_config") for log in ensemble_logs]

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

# ╔═╡ de7a5c20-d5d6-4fe5-b7c3-561b17a90143
function ensemble_init_model(ensemble_size, election, can_count, init_metrics, update_metrics, model_configs, diffusion_config)
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

# ╔═╡ b9ea7ab5-080b-496a-bac6-3d02cf8936b3
if model_source == "Load" 
	model, model_configs = load_model(model_dir_path)
else
	model, model_configs = General_model(voters, social_network, get_party_names(election), get_candidates(election))
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
# ╟─46e8650e-f57d-48d7-89de-1c72e12dea45
# ╟─bcc5468c-2a49-409d-b810-05fc30f4edca
# ╠═2f08ff39-d1d6-4341-8da8-5c46e23ae856
# ╠═c6ccf2a8-e045-4da9-bbdb-270327c2d53f
# ╟─c8522415-1fd9-4c06-bf2a-38ab23153b56
# ╠═61fa4389-3b1e-4be3-a5a2-ee37218d7d55
# ╟─977d39e2-7f82-49e8-a93f-889204bd19cb
# ╠═8ea22c93-1fe3-44b2-88c1-fb6ccd195866
# ╠═4aa4cdb4-af2a-4fcb-913d-389a9a899ab0
# ╟─100d0645-1178-428a-b4c1-3859ecb3ee18
# ╠═1626bf3e-8fe6-4e21-8cf4-76d84848114f
# ╠═8a760833-139b-406b-b985-a1daf7585ed3
# ╟─20893df2-fa42-46bf-889e-582b9ac39164
# ╟─831816fa-7983-4c05-9a76-a03631bf1e57
# ╠═40fdc182-c693-48fb-99ee-43d1bc78d95f
# ╠═342d3d34-4a3a-4f2d-811b-e9ab143504fd
# ╠═825b45d1-fe6a-4460-8fe2-b323019c56b6
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
# ╟─341a601f-2727-4362-b7d1-29dd47a92539
# ╟─759557c0-228f-4827-9038-1963b54a08d9
# ╠═637a053e-1ce8-4107-926a-46f46b4718f8
# ╟─345c5270-0822-4ba6-b326-14a60d8a8d46
# ╟─985131a7-7c11-4f9d-ae00-ef031002592d
# ╟─3cede6ac-5765-4c72-9b53-103c9c6a9bd9
# ╟─98885ec6-7561-43d7-bdf6-7f58fb2720f6
# ╠═72450aaf-c6c4-458e-9555-39c31345116b
# ╟─8e0750a9-2b83-4652-805b-dc1be2484161
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╠═9b6e56ed-cd89-4684-83d7-b2f931a07d14
# ╠═b9ea7ab5-080b-496a-bac6-3d02cf8936b3
# ╠═3671ba34-71bc-4e11-ad5e-aaeecf3f1e3f
# ╟─2c837f03-3329-4ab3-ad31-b0fe79df6bb7
# ╠═a13fff49-bc77-492c-8402-3fec14e9a7e4
# ╠═a6590191-0fbd-41a9-a7c3-b99e68bb27aa
# ╠═a652cf9b-adb5-47d2-906f-a7b479face45
# ╠═5bf9306d-ed0f-4d12-af1f-75ba9fe1e9a4
# ╠═988a9fcb-ad5f-4be6-b5f5-c3b05587a8d4
# ╠═4f7384d8-4c16-4a75-93ca-755215d74643
# ╟─09540198-bead-4df1-99bd-6e7848e7d215
# ╟─6f40b5c4-1252-472c-8932-11a2ee0935d2
# ╠═b6cd7a31-80b9-49eb-8004-34de5e6ad910
# ╠═e3f2c391-4fc3-47b5-bad2-5abc8f89a345
# ╠═e7f02f43-f8d5-4b7d-b097-4cbe7c7541a7
# ╠═970cf900-f459-4eee-a0d3-226a40b6422f
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╟─971693bb-1a08-4266-a93b-c3e9d60d8bcd
# ╟─de772425-25de-4228-b12e-d567b8ceb20f
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╟─aad0da61-59d2-4429-97fa-6612872bb863
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═e518a2db-a02e-4cfb-8587-f892fd9cbc85
# ╠═09d34d24-0fb0-4cc6-8ab6-c0d55b3346d0
# ╠═840c2562-c444-4422-9cf8-e82429163627
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
# ╠═48a2a821-f104-49da-b4e3-de7280f2eb03
# ╠═e8971acb-dc93-4a01-85f8-5190f1972ece
# ╠═5c505323-4487-4433-8ea7-4357b0fb8166
# ╠═a68fa858-2751-451d-a8f3-5c34c95e8b04
# ╠═f96c982d-b5db-47f7-91e0-9c9b3b36332f
# ╠═fc90baf0-a51d-4f3f-b036-b3bc381b2dbc
# ╟─4c140c0e-71c7-4567-b37e-286395a450a3
# ╟─f539cf71-34ae-4e22-a7ac-d259b55cb2d3
# ╠═d9fd7b27-7bd1-4e4d-a1b5-f4f173a86c2d
# ╠═1efa4e48-4a08-4aca-a255-8acd559c4bc8
# ╠═545103ec-9f9b-4b24-a993-818a20f18f49
# ╠═7f898ae7-8613-43a6-95e1-f61748cec34a
# ╠═dba61064-49ec-4ed7-b8f0-429751318282
# ╠═9b494231-4d23-4573-b686-e83119bafe88
# ╠═e35b5055-6ebd-49cd-83be-429f0f7dcaf5
# ╠═de7a5c20-d5d6-4fe5-b7c3-561b17a90143
# ╠═66267f11-eac9-4fbd-814f-89897f74a046
# ╠═eccddfc6-2e57-4f9e-88f3-88166fc5da11
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═8d259bcc-d54c-401f-b864-87701f2bcf46
# ╠═187da043-ee48-420a-91c7-5e3a4fdc30bb
# ╠═ff89eead-5bf5-4d52-9134-aa415ae156b7
# ╠═e70e36ad-066e-4619-a06a-56e325745a0e
# ╠═b94c9f37-b321-4db2-9da9-75917be8e52e
