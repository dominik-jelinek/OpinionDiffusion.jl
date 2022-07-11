### A Pluto.jl notebook ###
# v0.19.4

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

# ╔═╡ 957eb9d7-12d6-4a22-8338-4e8535b54c71
md"## Opinion diffusion"

# ╔═╡ d2923f02-66d7-47de-9801-d4ad99c1230f
md"### Intro to Pluto notebook"

# ╔═╡ 70e131de-4e31-4deb-9346-641e067269e3
md"Pluto is a reactive notebook that after the change of some variable recalculates all its dependencies. 
- Grey cell = executed cell
- Dark yellow cell = unsaved cell that was changed and is waiting for execution by you
- Red cell = cell that doesn't have all its dependencies or caused error
- I have put some execution barriers in form of checkmark to limit automatic execution of followed cells as the user might want to change variables first before time consuming model generation for example. It is still better to have reactive notebooks as the notebook is always in correct state. 
- At the time of writing this notebook it is possible to interrupt execution only on Linux.
- For faster restart of the notebook you can press Pluto.jl logo up on top and press black x right next to running notebook."

# ╔═╡ 581dab76-bc3e-48f5-8740-6e8cccdcc8d8
md"### Initialize packages"

# ╔═╡ 75588cfd-52c9-4406-975c-c03158db6e78
import Distributions, Distances, Graphs, Plots, Random

# ╔═╡ 55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
md"Choose visualization backend from Plots library. Plotly for interactivity. GR for speed."

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
Plots.plotly()
#Plots.gr()

# ╔═╡ 46e8650e-f57d-48d7-89de-1c72e12dea45
md"### Load dataset"

# ╔═╡ 470ec195-0118-4151-a74e-b976a91a3d29
#OpinionDiffusion.test_random_KT(1000, 8; log_lvl=0)

# ╔═╡ f31ef6e8-1edc-4e43-8cbc-cebab3455848
distance decreasing 1.29 distance decreasing unique 1.60

# ╔═╡ 63b5a286-8a8e-4f5e-8a33-c00ca03bf5d5
1.42 optimal mensie kroky

# ╔═╡ bcc5468c-2a49-409d-b810-05fc30f4edca
dataset = 2

# ╔═╡ fc5a5935-8f4e-47ad-8568-70cd61656e06
begin
	if dataset == 1
		input_filename = "ED-00001-00000001.toc"
		remove_candidates = [3, 5, 8, 11]
	elseif dataset == 2
		input_filename = "ED-00001-00000002.toc"
		remove_candidates = [8]#[1, 6, 8]
	elseif dataset == 3
		input_filename = "ED-00001-00000003.toc"
		remove_candidates = [3, 8, 9, 11]
	else
		input_filename = "madeUp.toc"
	end
end

# ╔═╡ c6ccf2a8-e045-4da9-bbdb-270327c2d53f
begin
	parties, src_candidates, src_election = parse_data(input_filename)
end

# ╔═╡ c8522415-1fd9-4c06-bf2a-38ab23153b56
md"### Explore dataset"

# ╔═╡ 21f341c3-4d38-4449-b0f8-62e4ab14c99b
filtered = [length(vote[end]) > 1 ? vote[1:end - 1] : vote  for vote in src_election ]

# ╔═╡ 042ec90d-8c0b-4acb-8d4c-31f701876cb6
function get_counts(votes, can_count)
	result = zeros(Int64, can_count, can_count)
	
	for vote in votes
        for (i, bucket) in enumerate(vote)
			#iterate buckets in vote
            result[bucket[1], i] += 1
        end
    end
	
	return result
end

# ╔═╡ 06d301e2-90c1-4fdf-b5ab-e127f8dee777
counts = get_counts(filtered, length(src_candidates))

# ╔═╡ d5d394c0-9b7c-4255-95fd-dc8cc32ca018
Plots.heatmap(counts, yticks=1:length(src_candidates), xticks=1:length(src_candidates), ylabel="Candidates", xlabel="Position")#[parties[can.party] for can in candidates])

# ╔═╡ 8c709e84-66f8-4128-a85c-66a4c5ffc9b7
Plots.bar(transpose(sum(counts, dims = 1)), legend=false, xticks=1:length(src_candidates), ylabel="# votes", xlabel="Vote position")

# ╔═╡ 20078431-5a7c-4a6f-b181-984ed54cf506
Plots.bar(sum(counts, dims = 2), legend=false, xticks=1:length(src_candidates), ylabel="# votes", xlabel="Candidate ID")

# ╔═╡ 977d39e2-7f82-49e8-a93f-889204bd19cb
md"### Sample voters"

# ╔═╡ 49230039-1716-4f48-b144-cddcb88e6172
g = OpinionDiffusion.get_DEG(voters_, dgrs, 0.1, ratio=1.0, log_lvl=false)

# ╔═╡ ad94415f-fc56-4a2f-9068-e05d8633aafe
init_sample_size = 1000

# ╔═╡ c50be339-8d1c-459f-8c14-934297f10717
function filter_candidates(election, candidates, remove_candidates, can_count)
	if length(remove_candidates) == 0
		return election, candidates
	end
	# calculate candidate index offset dependant 
	adjust = zeros(can_count)
	for i in 1:length(remove_candidates) - 1
		adjust[remove_candidates[i] + 1:remove_candidates[i + 1]-1] += fill(i, remove_candidates[i + 1] - remove_candidates[i] - 1)
	end
	adjust[remove_candidates[end] + 1:end] += fill(length(remove_candidates), can_count - remove_candidates[end])

	#copy election without the filtered out candidates
	new_election = Vector{Vector{Vector{Int64}}}()
	for vote in election
		new_vote = Vector{Vector{Int64}}()
		for bucket in vote
			new_bucket = Vector{Int64}()
			
			for can in bucket
				if can ∉ remove_candidates
					push!(new_bucket, can - adjust[can])
				end	
			end
			
			if length(new_bucket) != 0
				push!(new_vote, new_bucket)
			end
		end

		# vote with one bucket ore less buckets contains no preferences
		if length(new_vote) > 1
			push!(new_election, new_vote)
		end
	end

	new_candidates = Vector{OpinionDiffusion.Candidate}()
	for (i, can) in enumerate(candidates)
		if i ∉ remove_candidates
			push!(new_candidates, OpinionDiffusion.Candidate(can.name, can.party))
		end
	end
	
	#candidates = deleteat!(copy(candidates), remove_candidates)
	
	return new_election, new_candidates
end

# ╔═╡ 28f103a9-2b18-4cfc-bcf3-34d512f8da03
filtered_election, candidates = filter_candidates(src_election, src_candidates, remove_candidates, length(src_candidates))

# ╔═╡ 93b8909b-479c-422a-b475-2befadf5e9ec
election = filtered_election[OpinionDiffusion.StatsBase.sample(1:length(filtered_election), init_sample_size, replace=false)]

# ╔═╡ 150ecd7a-5fe5-4e25-9630-91d26c30ff38
src_candidates

# ╔═╡ 089c39a6-e516-4aac-8014-b3a1e6444e61
candidates

# ╔═╡ 20893df2-fa42-46bf-889e-582b9ac39164
md"### Configure model"

# ╔═╡ 831816fa-7983-4c05-9a76-a03631bf1e57
md"#### Voter config"

# ╔═╡ c84e5892-4a2e-4d26-92b7-d7720b188d4e
md"Kendall-tau based voter"

# ╔═╡ 519c9413-04ff-4a1e-995e-eaacf787df11
voter_config_kt = Kendall_voter_init_config(
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)

# ╔═╡ 40fdc182-c693-48fb-99ee-43d1bc78d95f
md"Spearman based voter"

# ╔═╡ d2abf148-7606-4ee2-826c-1aa2a26ab81d
begin
	weighting_rate = 2.0
	weight_func = position -> (length(candidates) - position)^weighting_rate
end

# ╔═╡ 342d3d34-4a3a-4f2d-811b-e9ab143504fd
begin
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

# ╔═╡ 64d43cc3-3142-4442-908a-f8bfb3077066
sum(weights)

# ╔═╡ 825b45d1-fe6a-4460-8fe2-b323019c56b6
Plots.bar([string(i) * '-' * string(i + 1) for i in 1:length(candidates)-1], [weight_func(x) for x in 1:length(candidates)-1], legend=false)

# ╔═╡ affcdfab-a84b-4bc1-860d-17b05ac18133
voter_config_sp = Spearman_voter_init_config(
		weights = weights, 
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)

# ╔═╡ c91395e7-393a-4ee1-ab44-7269cb1314d8
md"#### Graph config"

# ╔═╡ fe65fb44-faf6-44de-8e99-022c39ee3d33
graph_init_config_AB = weighted_AB_graph_config(
	m=5, 
	popularity_ratio=1.0)

# ╔═╡ 3b67247a-7dd8-49e7-ad88-64a8dad90d17
begin
	exp = 1.0
	scale = 2.0
	upper_limit = min(500, init_sample_size / 4)
	pareto = Distributions.truncated(Distributions.Pareto(exp, scale); upper=upper_limit)
	target_degree_distr = Int.(round.(rand(pareto, init_sample_size)))
end

# ╔═╡ 25374289-3cf1-4d27-887f-a47f9794f2bc
Plots.histogram(target_degree_distr)

# ╔═╡ 7069907d-2949-4f68-9602-9bef9ff46064
sum(target_degree_distr) / init_sample_size

# ╔═╡ 6c3f3dd3-8184-4aae-b40d-09830506773b
graph_init_config_DEG = DEG_graph_config(
	targed_deg_distr=target_degree_distr, 
	target_cc = 0.2, 
	ratio = 0.0,
	log_lvl =false
)

# ╔═╡ 338aa155-a4fb-4ff9-b27d-fc5481abb58d
md"#### Model config"

# ╔═╡ 0d3b58fe-8205-4d4c-9174-09cec37a61f3
model_config = General_model_config(
	voter_init_config = voter_config_sp,
	graph_init_config = graph_init_config_DEG
)

# ╔═╡ 3cede6ac-5765-4c72-9b53-103c9c6a9bd9
md"#### Create/load model"

# ╔═╡ b4d60582-ab23-4b2e-84bb-efd60786dc93
[file for file in readdir("logs")]

# ╔═╡ 637a053e-1ce8-4107-926a-46f46b4718f8
log_dir = "logs"

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir = log_dir * "/" * "model_2022-05-26_20-50-29"

# ╔═╡ 37cf8471-02a7-4a91-b17e-f44d06974d97
[file for file in readdir(model_dir)]

# ╔═╡ d8b613c5-7276-466d-a54a-7670f0921b35
exp_dir = model_dir * "/" * "experiment_2022-05-27_10-22-49"

# ╔═╡ a35b87d8-76eb-4b3e-b195-19dd5564bcf8
[exp_dir * "/" * file for file in readdir(exp_dir)]

# ╔═╡ 4712586c-bc93-43df-ae66-1e75a21b6f85
md"Index for loading specific model state inside of exp_dir. Insert -1 for the last state of the model."

# ╔═╡ 571e7a33-20b7-4432-b553-c07b9081c68d
idx = -1

# ╔═╡ e9169286-8d05-4bc9-8dc4-16ae6bd81038
logging = false

# ╔═╡ 98885ec6-7561-43d7-bdf6-7f58fb2720f6
md"Choose source of the model and then check execution barrier for generation of the model"

# ╔═╡ 72450aaf-c6c4-458e-9555-39c31345116b
md"""
Model source $(@bind model_source Select(["restart_model" => "Restart", "load_model" => "Load", "new_model" => "New (takes time)"]))
"""

# ╔═╡ 1937642e-7f63-4ffa-b01f-22208a716dac
md"""
Execution barrier $(@bind cb_model CheckBox())
"""

# ╔═╡ 36959890-84f4-4e67-92d4-a95fa909f693
Profile.print(format=:flat)

# ╔═╡ 2c1f81c6-a109-4da7-9cd6-7e761149f1e5
Profile.clear()

# ╔═╡ 93cb669f-6743-4b50-80de-c8594ea20497
if cb_model && logging
	if model_source == "new_model"
		model = General_model(election, length(candidates), model_config)
		logger = Logger(model)
		
	elseif model_source == "load_model" # load specific state and start new experiment
		model, logger = load_model(model_dir, exp_dir, idx, true)
		
	else # restart and create new experiment
		model, logger = restart_model(model_dir)
	end
elseif cb_model && !logging
	if model_source == "new_model"
		model = General_model(election, length(candidates), model_config)
		
	elseif model_source == "load_model"
		model = load_log(exp_dir, idx)
		
	else # restart
		model = load_log(model_dir)
	end
end

# ╔═╡ 776f99ea-6e44-4eb6-b2dd-3552bbb39954
logger.model_dir, logger.exp_dir

# ╔═╡ 5bf1a3d1-0daa-47b2-9fbc-4c1fcd3b4372
Graphs.global_clustering_coefficient(get_social_network(model))

# ╔═╡ 9609ec4b-ed07-4121-8db3-f7d374698ac0
sort(OpinionDiffusion.get_edge_distances2(model.social_network, model.voters), by=x -> x[3], rev=true)

# ╔═╡ fd0428ce-0c48-4727-a718-46f81e71d40f
begin
	frequencies = Dict()
	for vote in get_votes(get_voters(model))
		frequencies[vote] = get(frequencies, vote, 0) + 1
	end
end

# ╔═╡ 228b9f9c-1bde-4927-8c85-6b5938dcdbf1
collect(sort(frequencies, rev=true; byvalue=true))[1:min(length(frequencies), 10)]

# ╔═╡ 82b7bc72-ff9e-4881-9ce3-747261b947e9
Base.summarysize(get_votes(model))

# ╔═╡ 3465727f-b1bc-4e74-a447-438916003e35
Base.summarysize([Set([Set([Set([])])])])

# ╔═╡ 53586e20-27c6-4804-8d22-4b78ca6adf74
node_id = 42

# ╔═╡ eea97e32-d7b4-4789-8b51-a7a585770f5b
length(Graphs.neighbors(model.social_network, node_id))

# ╔═╡ ceadcd2d-42ac-4332-a49c-589cde3d500d
depth = 1.0

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

# ╔═╡ 045b7c19-a26d-4237-8a50-55abf7f9b57c
length(Graphs.neighbors(model.social_network, 1))

# ╔═╡ dd465b45-b378-4543-977d-909027d178d7
model.social_network

# ╔═╡ 596452b2-e74b-4a20-b3e0-3a918ffef042
egon = ego(model.social_network, 29000, 1)

# ╔═╡ 141d26c0-65f2-4266-b5a8-5b4b4c28f16e
#ego_project = reduce_dim(sampled_opinions, reduce_dim_config)[:, egon[2]]

# ╔═╡ bcf995a9-7f73-444c-adbc-fb13571112e9
x, y = ego_project[1, :], ego_project[2, :]

# ╔═╡ bb65c59a-5f29-4453-a5a5-dae5856347c0
length(Graphs.neighborhood(model.social_network, node_id, depth))

# ╔═╡ 52dc63c8-c864-4075-91d8-8c0a98a6a717
Graphs.induced_subgraph(model.social_network, Graphs.neighborhood(model.social_network, node_id, depth))

# ╔═╡ 09540198-bead-4df1-99bd-6e7848e7d215
md"### Play with diffusion"

# ╔═╡ 6f40b5c4-1252-472c-8932-11a2ee0935d2
md"Setup diffusion parameters and then check execution barrier for confirmation."

# ╔═╡ 0295587e-ad65-4bf1-b6d4-1b87a1e844ff
voter_diff_config_sp = Spearman_voter_diff_config(
			attract_proba = 0.7,
			change_rate = 0.5,
			normalize_shifts = (true, weights[1], weights[end])
        )

# ╔═╡ 63a4233d-c596-4844-8539-91a2223f2266
voter_diff_config_kt = Kendall_voter_diff_config(attract_proba = 0.8)

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diffusion_config = Diffusion_config(
        checkpoint = 1,
		evolve_vertices = 0.5,
		evolve_edges = 0.0,
        voter_diff_config = voter_diff_config_sp,
        graph_diff_config = General_graph_diff_config(
            popularity_ratio = 1.0
        )
    )

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
Execution barrier $(@bind cb_run CheckBox())
"""

# ╔═╡ d877c5d0-89af-48b9-bcd0-c1602d58339f
diffusions = 50

# ╔═╡ 27a60724-5d19-419f-b208-ffa0c78e2505
ensemble_size = 3

# ╔═╡ 87c573c1-69a4-4a61-bbb8-acb716f8ec6d
ensemble_model = false

# ╔═╡ aad0da61-59d2-4429-97fa-6612872bb863
md"### Diffusion analyzation"

# ╔═╡ a68fa858-2751-451d-a8f3-5c34c95e8b04
 ensemble_logs = [logger.model_dir * "/" * file for file in readdir(logger.model_dir) if split(file, "_")[1] == "ensemble"] 

# ╔═╡ 4add7dc5-0af8-4179-a496-a46767cc85ef
[
	"logs/model_2022-05-29_10-20-40/ensemble_2022-05-29_19-12-38.jld2",
	"logs/model_2022-05-29_22-40-22/ensemble_2022-05-29_22-40-51.jld2",
	"logs/model_2022-05-29_22-41-11/ensemble_2022-05-29_22-43-08.jld2"
]

# ╔═╡ 0932e410-4a74-42b3-86c5-ac40f6be3543
md"##### Visualization of voters in vector space defined by their opinions"

# ╔═╡ 563c5195-fd67-48d8-9b01-a73ea756a7ba
md"Sampling:"

# ╔═╡ 228f2d0b-b964-4133-b0bc-fee7d9fe298f
begin
	sample_size = max(256, ceil(Int, length(model.voters) * 0.01))
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)
end

# ╔═╡ 3fcf0afd-0bfc-4e58-b066-1461e49d1dee
sample_size

# ╔═╡ 9f7f56db-8a73-40f7-bb47-9570e41a634f
md"Load voters from model:"

# ╔═╡ b203aed5-3231-4e19-a961-089c0a4cf8c6
md"Dimensionality reduction for visualisation of high dimensional opinions"

# ╔═╡ 52893c8c-d1b5-482a-aae7-b3ec5c590b77
md"Clustering for colouring of voters based on their opinions"

# ╔═╡ 1adc5d59-6198-4ef5-9a8a-6390acc28be1
clustering_config = Clustering_config(
    used = true,
    #method = "K-means",
    method = "Party",
    kmeans_config = Kmeans_config(
        cluster_count = 5
    ),
    gm_config = GM_config(
    	cluster_count = 5
	)
)

# ╔═╡ fa927ac6-5d14-4a8d-8a50-16f305c00d11
labls, _ = clustering(voters_, candidates, length(parties), clustering_config)

# ╔═╡ a4e4ecb5-d28a-4ec3-93fd-ce58c402b6d3
OpinionDiffusion.drawGraph(g, labls, length(parties))

# ╔═╡ 0c2345e5-c1b2-4af8-ad31-617ddc99257a
md"##### Metrics calculated for every diffusion selected by slider below"

# ╔═╡ 86659fc0-af7e-4498-8388-3e79349e9eb4
md"""Diffusion: $(@bind step Slider(0 : logger.diff_counter[1], show_value=true, default=logger.diff_counter[1]))"""

# ╔═╡ 462930eb-f995-48f0-9564-0c3b3d3b437f
model_log = load_log(logger.exp_dir, step)

# ╔═╡ 05eedffd-c82a-4dbd-8608-5fa00f9d0bae
begin
	sampled_voters = get_voters(model_log)[sampled_voter_ids]
	sampled_opinions = reduce(hcat, get_opinion(sampled_voters))
end

# ╔═╡ a9ce1301-9a4c-46ab-9a6d-49109024d705
opinions = get_opinion(sampled_voters)

# ╔═╡ c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
labels, clusters = clustering(sampled_voters, candidates, length(parties), clustering_config)

# ╔═╡ b8f16963-914b-40e4-b13d-77cb6eb7b6db
reduce_dim_config = Reduce_dim_config(
    method = "PCA",
    pca_config = PCA_config(
        out_dim = 3
    ),
    tsne_config = Tsne_config(
        out_dim = 2,
        reduce_dims = 0,
        max_iter = 3000,
        perplexity = 100.0
    )
)

# ╔═╡ a652cf9b-adb5-47d2-906f-a7b479face45
OpinionDiffusion.model_vis(model, sampled_voter_ids, reduce_dim_config, clustering_config, parties, candidates)

# ╔═╡ e80cfc91-fc51-4216-a74c-39038d77c9db
begin
	_model_log = load_log(logger.exp_dir, 0)
	_sampled_opinions = reduce(hcat, get_opinion(_model_log.voters[sampled_voter_ids]))
	_projections = reduce_dim(_sampled_opinions, reduce_dim_config)
	
	x_projections = _projections[1, 1:length(candidates)]
	y_projections = _projections[2, 1:length(candidates)]
	
	min_x, max_x = minimum(_projections[1, :]), maximum(_projections[1, :]) 
	min_y, max_y = minimum(_projections[2, :]), maximum(_projections[2, :])
end

# ╔═╡ 008cbbe1-949d-4be7-9178-9bfa27c6e2a8
begin
	projections = reduce_dim(sampled_opinions, reduce_dim_config)
	#unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)
	projections
end

# ╔═╡ 83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
if step > 0
	prev_sampled_opinions = reduce(hcat, get_opinion(load_log(logger.exp_dir, step-1).voters[sampled_voter_ids]))
	difference = sampled_opinions - prev_sampled_opinions
	changes = vec(sum(abs.(difference), dims=1))
	println(sum(changes))
	draw_heat_vis(projections, changes, "Heat map")
end

# ╔═╡ f0eed2be-799f-46bb-968b-b70462cc06fd
title = reduce_dim_config.method * "_" * clustering_config.method * "_" * string(length(sampled_voters))

# ╔═╡ 18c07cb1-3a91-405c-a626-f608e0d1ea9d
OpinionDiffusion.GraphPlot.gplot(egon[1], x, y, nodesize=[Graphs.degree(model.social_network, node) for node in egon[2]]) #nodefillc=cluster_colors[labels[egon[2]]])

# ╔═╡ 4eb56ee4-04e9-40cd-90fc-1495215f2928
draw_voter_vis(projections, clusters, title)

# ╔═╡ f3dd8bf9-41d4-404a-8ab7-c3859be0de60
visualizations = OpinionDiffusion.gather_vis(logger.exp_dir, sampled_voter_ids, reduce_dim_config, clustering_config, parties, candidates)

# ╔═╡ 6415cf68-9648-412d-b6eb-9682ace5c9ec
md"""Diffusion: $(@bind t Slider(0 : logger.diff_counter[1], show_value=true, default=logger.diff_counter[1]))"""

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
plots = Plots.plot([Plots.heatmap(count, yticks=1:length(src_candidates), xticks=1:length(src_candidates), ylabel="Candidate", xlabel="Position") for count in countss]..., layout = (5, 1), size = (669,900))

# ╔═╡ e814359f-3e70-48ab-9b57-aed8bcaf0102
clusters

# ╔═╡ cd8ffe9d-e990-43ed-ad77-28067d1536ed
draw_degree_distr(Graphs.degree_histogram(model_log.social_network))

# ╔═╡ d716423e-7945-4e0a-a6ab-17e0b94c721e
md"##### Compounded metrics for all diffusions"

# ╔═╡ 88ffc9b6-327b-4c6f-827b-67aa7e175855
candidates

# ╔═╡ d0193993-832d-412d-8a3a-3b804b0845c7
src_candidates

# ╔═╡ ae6f392e-6920-4e88-8587-837f03a00a88
parties

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
    metrics["clustering_coeff"] = [Graphs.global_clustering_coefficient(g)]

    #election results
	votes = get_votes(voters)
	metrics["avg_vote_length"] = [OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes])]
    metrics["mean_nei_dist"] = [OpinionDiffusion.StatsBase.mean([OpinionDiffusion.StatsBase.mean(get_distance(voter, voters[Graphs.neighbors(g, voter.ID)])) for voter in voters])]
    metrics["unique_votes"] = [length(unique(votes))]

    metrics["plurality_votings"] = [plurality_voting(votes, can_count, true)]
    metrics["borda_votings"] = [borda_voting(votes, can_count, true)]
    metrics["copeland_votings"] = [copeland_voting(votes, can_count)]

	metrics["positions"] = [get_positions(voters, can_count)]
	
	return metrics
end

# ╔═╡ 109e1b7c-16f0-4450-87ea-2d0442df5589
metrics = init_metrics(model, length(candidates))

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
    push!(diffusion_metrics["clustering_coeff"], Graphs.global_clustering_coefficient(g))
    
    votes = get_votes(voters)
	push!(diffusion_metrics["avg_vote_length"], OpinionDiffusion.StatsBase.mean([length(vote) for vote in votes]))
    
    push!(diffusion_metrics["mean_nei_dist"], OpinionDiffusion.StatsBase.mean([OpinionDiffusion.StatsBase.mean(get_distance(voter, voters[Graphs.neighbors(g, voter.ID)])) for voter in voters]))
    push!(diffusion_metrics["unique_votes"], length(unique(votes)))

    push!(diffusion_metrics["plurality_votings"], plurality_voting(votes, can_count, true))
    push!(diffusion_metrics["borda_votings"], borda_voting(votes, can_count, true))
    push!(diffusion_metrics["copeland_votings"], copeland_voting(votes, can_count))
    push!(diffusion_metrics["positions"], get_positions(voters, can_count))
end

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_run
	if ensemble_size == 1
		for i in 1:diffusions
			run!(model, diffusion_config, logger)
			update_metrics!(model, metrics, length(candidates))
		end
	elseif ensemble_model
		gathered_metrics = OpinionDiffusion.run_ensemble_model(ensemble_size, diffusions, election, init_metrics, length(candidates), update_metrics!, model_config, diffusion_config, true)
	else
		gathered_metrics = OpinionDiffusion.run_ensemble(model, ensemble_size, diffusions, metrics, update_metrics!, diffusion_config, logger.model_dir)
	end
end

# ╔═╡ 187da043-ee48-420a-91c7-5e3a4fdc30bb
function draw_voting_rules(data, voting_rules, candidates, parties)
	n = length(voting_rules)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)

	for (i, metric) in enumerate(voting_rules)
		result = transpose(reduce(hcat, data[metric]))
		OpinionDiffusion.draw_voting_res!(plot[i, 1], candidates, parties, result, metric)
	end
	
	Plots.plot(plot)
end

# ╔═╡ 840c2562-c444-4422-9cf8-e82429163627
draw_voting_rules(gathered_metrics, ["plurality_votings", "borda_votings", "copeland_votings", "positions"], candidates, parties)

# ╔═╡ 5c505323-4487-4433-8ea7-4357b0fb8166
configss = [OpinionDiffusion.load(log, "diffusion_config") for log in ensemble_logs]

# ╔═╡ 3ef60b95-f1fd-44ab-a68e-499e4e41b832
OpinionDiffusion.load(ensemble_logs[3], "diffusion_config")

# ╔═╡ 03c10533-390f-4bce-a4e7-7df3e35146ff
show(configss[1])

# ╔═╡ 5f799588-87ab-4030-b310-62ce351ef2b2
methods(dump)

# ╔═╡ c8d407f1-6ed1-4007-8a46-3dd37664e4d6
dump(configss[1])

# ╔═╡ ff89eead-5bf5-4d52-9134-aa415ae156b7
function compare_voting_rules(logs, voting_rules, candidates, parties)
	data = [OpinionDiffusion.load(log, "gathered_metrics") for log in logs]
	
	n = length(voting_rules)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)

	
	for (i, metric) in enumerate(voting_rules)
		for (j, dato) in enumerate(data)
			result = transpose(reduce(hcat, dato[metric]))
			OpinionDiffusion.draw_voting_res!(plot[i, 1], candidates, parties, result, metric, log_idx=string(j))
		end
	end
	
	Plots.plot(plot)
end

# ╔═╡ fc90baf0-a51d-4f3f-b036-b3bc381b2dbc
compare_voting_rules(ensemble_logs, ["plurality_votings", "borda_votings", "copeland_votings"], candidates, parties)

# ╔═╡ e70e36ad-066e-4619-a06a-56e325745a0e
function draw_metrics_vis(data, metrics)
	n = length(metrics)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)

	for (i, metric) in enumerate(metrics)
		OpinionDiffusion.draw_metric!(plot[i, 1], data[metric], metric)
	end

	Plots.plot(plot)
end

# ╔═╡ 09d34d24-0fb0-4cc6-8ab6-c0d55b3346d0
draw_metrics_vis(gathered_metrics, ["unique_votes", "avg_vote_length", "mean_nei_dist", "avg_degrees"])

# ╔═╡ b94c9f37-b321-4db2-9da9-75917be8e52e
function compare_metrics_vis(logs, metrics)
	data = [OpinionDiffusion.load(log, "gathered_metrics") for log in logs]
	
	n = length(metrics)
	plot = Plots.plot(size = Plots.default(:size) .* (1, n), layout = (n, 1), bottom_margin = 10Plots.mm, left_margin = 5Plots.mm)
	
	for (i, metric) in enumerate(metrics)
		for (j, dato) in enumerate(data)
			OpinionDiffusion.draw_metric!(plot[i, 1], dato[metric], metric, log_idx=j)
		end
	end

	Plots.plot(plot)
end

# ╔═╡ f96c982d-b5db-47f7-91e0-9c9b3b36332f
compare_metrics_vis(ensemble_logs, ["unique_votes", "avg_vote_length", "mean_nei_dist", "avg_degrees"])

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
# ╠═470ec195-0118-4151-a74e-b976a91a3d29
# ╠═f31ef6e8-1edc-4e43-8cbc-cebab3455848
# ╠═63b5a286-8a8e-4f5e-8a33-c00ca03bf5d5
# ╠═bcc5468c-2a49-409d-b810-05fc30f4edca
# ╠═fc5a5935-8f4e-47ad-8568-70cd61656e06
# ╠═c6ccf2a8-e045-4da9-bbdb-270327c2d53f
# ╟─c8522415-1fd9-4c06-bf2a-38ab23153b56
# ╠═21f341c3-4d38-4449-b0f8-62e4ab14c99b
# ╠═042ec90d-8c0b-4acb-8d4c-31f701876cb6
# ╠═06d301e2-90c1-4fdf-b5ab-e127f8dee777
# ╠═d5d394c0-9b7c-4255-95fd-dc8cc32ca018
# ╠═8c709e84-66f8-4128-a85c-66a4c5ffc9b7
# ╠═20078431-5a7c-4a6f-b181-984ed54cf506
# ╟─977d39e2-7f82-49e8-a93f-889204bd19cb
# ╠═49230039-1716-4f48-b144-cddcb88e6172
# ╠═fa927ac6-5d14-4a8d-8a50-16f305c00d11
# ╠═a4e4ecb5-d28a-4ec3-93fd-ce58c402b6d3
# ╠═ad94415f-fc56-4a2f-9068-e05d8633aafe
# ╠═93b8909b-479c-422a-b475-2befadf5e9ec
# ╠═c50be339-8d1c-459f-8c14-934297f10717
# ╠═28f103a9-2b18-4cfc-bcf3-34d512f8da03
# ╠═150ecd7a-5fe5-4e25-9630-91d26c30ff38
# ╠═089c39a6-e516-4aac-8014-b3a1e6444e61
# ╟─20893df2-fa42-46bf-889e-582b9ac39164
# ╟─831816fa-7983-4c05-9a76-a03631bf1e57
# ╟─c84e5892-4a2e-4d26-92b7-d7720b188d4e
# ╠═519c9413-04ff-4a1e-995e-eaacf787df11
# ╟─40fdc182-c693-48fb-99ee-43d1bc78d95f
# ╠═d2abf148-7606-4ee2-826c-1aa2a26ab81d
# ╠═342d3d34-4a3a-4f2d-811b-e9ab143504fd
# ╠═c2d75170-789d-41b4-b30b-40b143d23702
# ╠═64d43cc3-3142-4442-908a-f8bfb3077066
# ╠═825b45d1-fe6a-4460-8fe2-b323019c56b6
# ╠═affcdfab-a84b-4bc1-860d-17b05ac18133
# ╟─c91395e7-393a-4ee1-ab44-7269cb1314d8
# ╠═fe65fb44-faf6-44de-8e99-022c39ee3d33
# ╠═3b67247a-7dd8-49e7-ad88-64a8dad90d17
# ╠═25374289-3cf1-4d27-887f-a47f9794f2bc
# ╠═7069907d-2949-4f68-9602-9bef9ff46064
# ╠═6c3f3dd3-8184-4aae-b40d-09830506773b
# ╟─338aa155-a4fb-4ff9-b27d-fc5481abb58d
# ╠═0d3b58fe-8205-4d4c-9174-09cec37a61f3
# ╟─3cede6ac-5765-4c72-9b53-103c9c6a9bd9
# ╠═776f99ea-6e44-4eb6-b2dd-3552bbb39954
# ╠═b4d60582-ab23-4b2e-84bb-efd60786dc93
# ╠═637a053e-1ce8-4107-926a-46f46b4718f8
# ╠═985131a7-7c11-4f9d-ae00-ef031002592d
# ╠═37cf8471-02a7-4a91-b17e-f44d06974d97
# ╠═d8b613c5-7276-466d-a54a-7670f0921b35
# ╠═a35b87d8-76eb-4b3e-b195-19dd5564bcf8
# ╟─4712586c-bc93-43df-ae66-1e75a21b6f85
# ╠═571e7a33-20b7-4432-b553-c07b9081c68d
# ╠═e9169286-8d05-4bc9-8dc4-16ae6bd81038
# ╟─98885ec6-7561-43d7-bdf6-7f58fb2720f6
# ╟─72450aaf-c6c4-458e-9555-39c31345116b
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╠═36959890-84f4-4e67-92d4-a95fa909f693
# ╠═2c1f81c6-a109-4da7-9cd6-7e761149f1e5
# ╠═93cb669f-6743-4b50-80de-c8594ea20497
# ╠═5bf1a3d1-0daa-47b2-9fbc-4c1fcd3b4372
# ╠═a652cf9b-adb5-47d2-906f-a7b479face45
# ╠═9609ec4b-ed07-4121-8db3-f7d374698ac0
# ╠═fd0428ce-0c48-4727-a718-46f81e71d40f
# ╠═228b9f9c-1bde-4927-8c85-6b5938dcdbf1
# ╠═82b7bc72-ff9e-4881-9ce3-747261b947e9
# ╠═3465727f-b1bc-4e74-a447-438916003e35
# ╠═53586e20-27c6-4804-8d22-4b78ca6adf74
# ╠═eea97e32-d7b4-4789-8b51-a7a585770f5b
# ╠═ceadcd2d-42ac-4332-a49c-589cde3d500d
# ╠═5a792e2e-909b-4690-a0bf-d05afe7f4c81
# ╠═045b7c19-a26d-4237-8a50-55abf7f9b57c
# ╠═dd465b45-b378-4543-977d-909027d178d7
# ╠═596452b2-e74b-4a20-b3e0-3a918ffef042
# ╠═141d26c0-65f2-4266-b5a8-5b4b4c28f16e
# ╠═bcf995a9-7f73-444c-adbc-fb13571112e9
# ╠═bb65c59a-5f29-4453-a5a5-dae5856347c0
# ╠═52dc63c8-c864-4075-91d8-8c0a98a6a717
# ╟─09540198-bead-4df1-99bd-6e7848e7d215
# ╟─6f40b5c4-1252-472c-8932-11a2ee0935d2
# ╠═0295587e-ad65-4bf1-b6d4-1b87a1e844ff
# ╠═63a4233d-c596-4844-8539-91a2223f2266
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═d877c5d0-89af-48b9-bcd0-c1602d58339f
# ╠═27a60724-5d19-419f-b208-ffa0c78e2505
# ╠═87c573c1-69a4-4a61-bbb8-acb716f8ec6d
# ╠═109e1b7c-16f0-4450-87ea-2d0442df5589
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╟─aad0da61-59d2-4429-97fa-6612872bb863
# ╠═09d34d24-0fb0-4cc6-8ab6-c0d55b3346d0
# ╠═a68fa858-2751-451d-a8f3-5c34c95e8b04
# ╠═4add7dc5-0af8-4179-a496-a46767cc85ef
# ╠═f96c982d-b5db-47f7-91e0-9c9b3b36332f
# ╠═840c2562-c444-4422-9cf8-e82429163627
# ╠═fc90baf0-a51d-4f3f-b036-b3bc381b2dbc
# ╟─0932e410-4a74-42b3-86c5-ac40f6be3543
# ╟─563c5195-fd67-48d8-9b01-a73ea756a7ba
# ╠═228f2d0b-b964-4133-b0bc-fee7d9fe298f
# ╠═3fcf0afd-0bfc-4e58-b066-1461e49d1dee
# ╠═462930eb-f995-48f0-9564-0c3b3d3b437f
# ╟─9f7f56db-8a73-40f7-bb47-9570e41a634f
# ╠═05eedffd-c82a-4dbd-8608-5fa00f9d0bae
# ╟─b203aed5-3231-4e19-a961-089c0a4cf8c6
# ╠═e80cfc91-fc51-4216-a74c-39038d77c9db
# ╠═008cbbe1-949d-4be7-9178-9bfa27c6e2a8
# ╟─52893c8c-d1b5-482a-aae7-b3ec5c590b77
# ╠═1adc5d59-6198-4ef5-9a8a-6390acc28be1
# ╠═a9ce1301-9a4c-46ab-9a6d-49109024d705
# ╠═c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
# ╠═f0eed2be-799f-46bb-968b-b70462cc06fd
# ╟─0c2345e5-c1b2-4af8-ad31-617ddc99257a
# ╠═86659fc0-af7e-4498-8388-3e79349e9eb4
# ╠═83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
# ╠═b8f16963-914b-40e4-b13d-77cb6eb7b6db
# ╠═18c07cb1-3a91-405c-a626-f608e0d1ea9d
# ╠═4eb56ee4-04e9-40cd-90fc-1495215f2928
# ╠═7e47b5c8-3ad1-4093-85e5-6b20238e9569
# ╠═f3dd8bf9-41d4-404a-8ab7-c3859be0de60
# ╠═6415cf68-9648-412d-b6eb-9682ace5c9ec
# ╠═9b8fc5b4-9918-4c6b-8389-b19d5faf16fc
# ╠═53b63024-6f60-4121-971d-50a6205c05a8
# ╠═c7386f6d-451c-49d7-9fd4-1277b734f7f0
# ╠═a1681b3e-340f-4c8a-a991-98c8217a05ad
# ╠═b771f5b9-42e1-444f-b5ac-cba2f0162e8d
# ╠═7bff8689-580e-43b5-9614-65b6f482eb5c
# ╠═e814359f-3e70-48ab-9b57-aed8bcaf0102
# ╠═cd8ffe9d-e990-43ed-ad77-28067d1536ed
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═88ffc9b6-327b-4c6f-827b-67aa7e175855
# ╠═d0193993-832d-412d-8a3a-3b804b0845c7
# ╠═ae6f392e-6920-4e88-8587-837f03a00a88
# ╟─4c140c0e-71c7-4567-b37e-286395a450a3
# ╟─f539cf71-34ae-4e22-a7ac-d259b55cb2d3
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═e31cdd5b-3310-43fb-addc-96144547de2b
# ╠═8d259bcc-d54c-401f-b864-87701f2bcf46
# ╠═187da043-ee48-420a-91c7-5e3a4fdc30bb
# ╠═5c505323-4487-4433-8ea7-4357b0fb8166
# ╠═3ef60b95-f1fd-44ab-a68e-499e4e41b832
# ╠═03c10533-390f-4bce-a4e7-7df3e35146ff
# ╠═5f799588-87ab-4030-b310-62ce351ef2b2
# ╠═c8d407f1-6ed1-4007-8a46-3dd37664e4d6
# ╠═ff89eead-5bf5-4d52-9134-aa415ae156b7
# ╠═e70e36ad-066e-4619-a06a-56e325745a0e
# ╠═b94c9f37-b321-4db2-9da9-75917be8e52e
