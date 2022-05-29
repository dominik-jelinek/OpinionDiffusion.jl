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
import Distributions, Distances, Graphs, Plots

# ╔═╡ 46e8650e-f57d-48d7-89de-1c72e12dea45
md"### Load dataset"

# ╔═╡ bcc5468c-2a49-409d-b810-05fc30f4edca
dataset = 2

# ╔═╡ fc5a5935-8f4e-47ad-8568-70cd61656e06
begin
	if dataset == 1
		input_filename = "ED-00001-00000001.toc"
		remove_candidates = [3, 5, 8, 11]
	elseif dataset == 2
		input_filename = "ED-00001-00000002.toc"
		remove_candidates = [1, 6, 8]
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
Plots.bar(sum(counts, dims = 2), legend=false, xticks=1:length(src_candidates), ylabel="# votes", xlabel="candidate ID")

# ╔═╡ dfef082a-3615-4141-9852-693de7ad4255
init_sample_size = 1000

# ╔═╡ c50be339-8d1c-459f-8c14-934297f10717
function filter_candidates(election, candidates, remove_candidates, can_count)
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
md"### Setup model"

# ╔═╡ c84e5892-4a2e-4d26-92b7-d7720b188d4e
md"Kendall-tau based voter"

# ╔═╡ 519c9413-04ff-4a1e-995e-eaacf787df11
voter_config_kt = Kendall_voter_init_config(
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)

# ╔═╡ 40fdc182-c693-48fb-99ee-43d1bc78d95f
md"Spearman based voter"

# ╔═╡ 342d3d34-4a3a-4f2d-811b-e9ab143504fd
weighting_rate = 1.0

# ╔═╡ affcdfab-a84b-4bc1-860d-17b05ac18133
voter_config_sp = Spearman_voter_init_config(
		weight_func = position -> (length(candidates) - position)^weighting_rate, 
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)

# ╔═╡ 825b45d1-fe6a-4460-8fe2-b323019c56b6
Plots.bar([string(i) * '-' * string(i + 1) for i in 1:length(candidates)-1], [voter_config_sp.weight_func(x) for x in 1:length(candidates)-1], legend=false)

# ╔═╡ 338aa155-a4fb-4ff9-b27d-fc5481abb58d
md"Configuration of general model"

# ╔═╡ 0d3b58fe-8205-4d4c-9174-09cec37a61f3
model_config = General_model_config( 
	m = 32,
	popularity_ratio = 0.5,
	voter_init_config = voter_config_sp
)

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir = "logs/" * "model_2022-05-26_20-50-29"

# ╔═╡ d8b613c5-7276-466d-a54a-7670f0921b35
exp_dir = model_dir * "/" * "experiment_2022-04-29_18-42-43"

# ╔═╡ 4712586c-bc93-43df-ae66-1e75a21b6f85
md"Index for loading specific model state inside of exp_dir. Insert -1 for the last state of the model."

# ╔═╡ 571e7a33-20b7-4432-b553-c07b9081c68d
idx = -1

# ╔═╡ e9169286-8d05-4bc9-8dc4-16ae6bd81038
logging = true

# ╔═╡ 698861ff-f8a8-4fb6-a31d-6a5ac8607aaf
length(candidates)

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
Graphs.global_clustering_coefficient(model.social_network)

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
			attract_proba = 0.8,
			change_rate = 0.5,
			normalize_shifts = (true, model_config.voter_init_config.weight_func(length(candidates)), model_config.voter_init_config.weight_func(1))
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
            dist_metric = Distances.Cityblock(),
            edge_diff_func = dist -> (1/2)^(dist)
        )
    )

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
Execution barrier $(@bind cb_run CheckBox())
"""

# ╔═╡ d877c5d0-89af-48b9-bcd0-c1602d58339f
diffusions = 40

# ╔═╡ 27a60724-5d19-419f-b208-ffa0c78e2505
ensemble_size = 2

# ╔═╡ 2679d25d-aaf4-4892-b7b8-668ff1ee9811
readdir(logger.model_dir)

# ╔═╡ 7cbfaec0-3d1f-4262-9eca-281723d1467e


# ╔═╡ 73cbe526-02ca-4185-9484-041e3dc0166c


# ╔═╡ 0bf0d9c9-9e82-4305-9bb5-0dc8f2dcf107
OpinionDiffusion.load()

# ╔═╡ 6492a611-fe57-4279-8852-9271b91396cc
Profile.print(format=:flat)

# ╔═╡ 260af73d-28de-46da-8f80-54f4349e6fba
Profile.clear()

# ╔═╡ aad0da61-59d2-4429-97fa-6612872bb863
md"### Diffusion analyzation"

# ╔═╡ 55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
md"Choose visualization backend from Plots library:"

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
Plots.plotly()
#Plots.gr()

# ╔═╡ a68fa858-2751-451d-a8f3-5c34c95e8b04
ensemble_logs = [logger.model_dir * "/" * file for file in readdir(logger.model_dir) if split(file, "_")[1] == "ensemble"] 

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
        out_dim = 2
    ),
    tsne_config = Tsne_config(
        out_dim = 2,
        reduce_dims = 0,
        max_iter = 3000,
        perplexity = 100.0
    )
)

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

# ╔═╡ 9a049ea3-f5c6-4e69-ab71-c2a8b8bad178


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

# ╔═╡ d6c3ae90-2890-488f-8d76-d668236c0dc9
draw_edge_distances(get_edge_distances(model_log.social_network, model_log.voters))

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
end

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_run && ensemble_size == 1
	for i in 1:diffusions
		run!(model, diffusion_config, logger)
		update_metrics!(model, metrics, length(candidates))
	end
elseif cb_run
	ens_metrics = OpinionDiffusion.run_ensemble!(model, ensemble_size, diffusions, metrics, update_metrics!, diffusion_config)
	gathered_metrics = OpinionDiffusion.gather_metrics(ens_metrics)

	if logging
		save_ensemble(logger.model_dir, diffusion_config, gathered_metrics)
	end
end

# ╔═╡ 8fec2372-6817-4c96-8884-5757ab851f44
ens_metrics

# ╔═╡ 009b8bbb-1419-4ded-bf6d-add522ed89d7
gathered_metrics

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
draw_voting_rules(gathered_metrics, ["plurality_votings", "borda_votings", "copeland_votings"], candidates, parties)

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

# ╔═╡ f6126696-ab8d-4637-b815-dd89ae8fb171
function metrics_vis(metrics, candidates, parties, exp_dir=Nothing)
    degrees = draw_range(metrics["min_degrees"], metrics["avg_degrees"], metrics["max_degrees"], title="Degree range", xlabel="Diffusions", ylabel="Degree", value_label="avg")

    plurality = draw_voting_res(candidates, parties, reduce(hcat, metrics["plurality_votings"])', "Plurality voting")
    borda = draw_voting_res(candidates, parties, reduce(hcat, metrics["borda_votings"])', "Borda voting")
    copeland = draw_voting_res(candidates, parties, reduce(hcat, metrics["copeland_votings"])', "Copeland voting")

    plots = Plots.plot(degrees, plurality, borda, copeland, layout = (2, 2), size = (669,900))
    
    if exp_dir != Nothing
        Plots.savefig(plots, "$(exp_dir)/images/metrics.png")
    end

    return plots
end

# ╔═╡ 7f138d72-419a-4642-b163-6ec58ce42d24
metrics_vis(metrics, candidates, parties)

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
# ╟─46e8650e-f57d-48d7-89de-1c72e12dea45
# ╠═bcc5468c-2a49-409d-b810-05fc30f4edca
# ╠═fc5a5935-8f4e-47ad-8568-70cd61656e06
# ╠═c6ccf2a8-e045-4da9-bbdb-270327c2d53f
# ╠═21f341c3-4d38-4449-b0f8-62e4ab14c99b
# ╠═042ec90d-8c0b-4acb-8d4c-31f701876cb6
# ╠═06d301e2-90c1-4fdf-b5ab-e127f8dee777
# ╠═d5d394c0-9b7c-4255-95fd-dc8cc32ca018
# ╠═8c709e84-66f8-4128-a85c-66a4c5ffc9b7
# ╠═20078431-5a7c-4a6f-b181-984ed54cf506
# ╠═28f103a9-2b18-4cfc-bcf3-34d512f8da03
# ╠═dfef082a-3615-4141-9852-693de7ad4255
# ╠═93b8909b-479c-422a-b475-2befadf5e9ec
# ╠═c50be339-8d1c-459f-8c14-934297f10717
# ╠═150ecd7a-5fe5-4e25-9630-91d26c30ff38
# ╠═089c39a6-e516-4aac-8014-b3a1e6444e61
# ╟─20893df2-fa42-46bf-889e-582b9ac39164
# ╟─c84e5892-4a2e-4d26-92b7-d7720b188d4e
# ╠═519c9413-04ff-4a1e-995e-eaacf787df11
# ╟─40fdc182-c693-48fb-99ee-43d1bc78d95f
# ╠═342d3d34-4a3a-4f2d-811b-e9ab143504fd
# ╠═affcdfab-a84b-4bc1-860d-17b05ac18133
# ╠═825b45d1-fe6a-4460-8fe2-b323019c56b6
# ╟─338aa155-a4fb-4ff9-b27d-fc5481abb58d
# ╠═0d3b58fe-8205-4d4c-9174-09cec37a61f3
# ╠═776f99ea-6e44-4eb6-b2dd-3552bbb39954
# ╠═985131a7-7c11-4f9d-ae00-ef031002592d
# ╠═d8b613c5-7276-466d-a54a-7670f0921b35
# ╟─4712586c-bc93-43df-ae66-1e75a21b6f85
# ╠═571e7a33-20b7-4432-b553-c07b9081c68d
# ╠═e9169286-8d05-4bc9-8dc4-16ae6bd81038
# ╠═698861ff-f8a8-4fb6-a31d-6a5ac8607aaf
# ╟─98885ec6-7561-43d7-bdf6-7f58fb2720f6
# ╟─72450aaf-c6c4-458e-9555-39c31345116b
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╠═93cb669f-6743-4b50-80de-c8594ea20497
# ╠═5bf1a3d1-0daa-47b2-9fbc-4c1fcd3b4372
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
# ╠═109e1b7c-16f0-4450-87ea-2d0442df5589
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═2679d25d-aaf4-4892-b7b8-668ff1ee9811
# ╠═7cbfaec0-3d1f-4262-9eca-281723d1467e
# ╠═73cbe526-02ca-4185-9484-041e3dc0166c
# ╠═0bf0d9c9-9e82-4305-9bb5-0dc8f2dcf107
# ╠═8fec2372-6817-4c96-8884-5757ab851f44
# ╠═009b8bbb-1419-4ded-bf6d-add522ed89d7
# ╠═6492a611-fe57-4279-8852-9271b91396cc
# ╠═260af73d-28de-46da-8f80-54f4349e6fba
# ╟─aad0da61-59d2-4429-97fa-6612872bb863
# ╟─55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
# ╠═ff873978-d93f-4ba2-aadd-6cfd3b136e3d
# ╠═09d34d24-0fb0-4cc6-8ab6-c0d55b3346d0
# ╠═a68fa858-2751-451d-a8f3-5c34c95e8b04
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
# ╠═9a049ea3-f5c6-4e69-ab71-c2a8b8bad178
# ╠═b771f5b9-42e1-444f-b5ac-cba2f0162e8d
# ╠═7bff8689-580e-43b5-9614-65b6f482eb5c
# ╠═e814359f-3e70-48ab-9b57-aed8bcaf0102
# ╟─cd8ffe9d-e990-43ed-ad77-28067d1536ed
# ╠═d6c3ae90-2890-488f-8d76-d668236c0dc9
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═7f138d72-419a-4642-b163-6ec58ce42d24
# ╠═88ffc9b6-327b-4c6f-827b-67aa7e175855
# ╠═d0193993-832d-412d-8a3a-3b804b0845c7
# ╠═ae6f392e-6920-4e88-8587-837f03a00a88
# ╠═4c140c0e-71c7-4567-b37e-286395a450a3
# ╟─f539cf71-34ae-4e22-a7ac-d259b55cb2d3
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═e31cdd5b-3310-43fb-addc-96144547de2b
# ╠═8d259bcc-d54c-401f-b864-87701f2bcf46
# ╠═187da043-ee48-420a-91c7-5e3a4fdc30bb
# ╠═ff89eead-5bf5-4d52-9134-aa415ae156b7
# ╠═e70e36ad-066e-4619-a06a-56e325745a0e
# ╠═b94c9f37-b321-4db2-9da9-75917be8e52e
# ╠═f6126696-ab8d-4637-b815-dd89ae8fb171
