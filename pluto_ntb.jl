### A Pluto.jl notebook ###
# v0.17.1

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
using Revise, PlutoUI, OpinionDiffusion

# ╔═╡ c5aa2487-97aa-48bd-b357-4e806a4c41e9
using Profile

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

# ╔═╡ fc5a5935-8f4e-47ad-8568-70cd61656e06
input_filename = "ED-00001-00000002.toc"
#input_filename = "madeUp.toc"

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
Plots.bar(transpose(sum(counts, dims = 1)), legend=false, xticks=1:9, ylabel="# votes", xlabel="Vote position")

# ╔═╡ 20078431-5a7c-4a6f-b181-984ed54cf506
Plots.bar(sum(counts, dims = 2), legend=false, xticks=1:9, ylabel="# votes", xlabel="candidate ID")

# ╔═╡ 70642f15-ac19-4a9c-993c-b567d336b43b
remove_candidates = [1, 6, 8]

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
election, candidates = filter_candidates(src_election, src_candidates, remove_candidates, length(src_candidates))

# ╔═╡ 150ecd7a-5fe5-4e25-9630-91d26c30ff38
src_candidates

# ╔═╡ 089c39a6-e516-4aac-8014-b3a1e6444e61
candidates

# ╔═╡ 20893df2-fa42-46bf-889e-582b9ac39164
md"### Setup model"

# ╔═╡ 519c9413-04ff-4a1e-995e-eaacf787df11
voter_config_kt = Kendall_voter_init_config(
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)

# ╔═╡ affcdfab-a84b-4bc1-860d-17b05ac18133
voter_config_sp = Spearman_voter_init_config(
		weight_func = position -> (1.0/2.0)^(position), 
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)

# ╔═╡ 0d3b58fe-8205-4d4c-9174-09cec37a61f3
model_config = General_model_config( 
	m = 32,
	popularity_ratio = 0.5,
	voter_config = voter_config_sp
)

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir = "logs/" * "model_2022-04-12_16-38-26"

# ╔═╡ d8b613c5-7276-466d-a54a-7670f0921b35
exp_dir = model_dir * "/" * "experiment_2021-11-07_20-35-37"

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
node_id = 10001

# ╔═╡ eea97e32-d7b4-4789-8b51-a7a585770f5b
length(Graphs.neighbors(model.social_network, node_id))

# ╔═╡ ceadcd2d-42ac-4332-a49c-589cde3d500d
depth = 0.5

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
#ego_project = reduce_dim(get_opinion(model.voters), reduce_dim_config)[:, egon[2]]

# ╔═╡ bcf995a9-7f73-444c-adbc-fb13571112e9
x, y = ego_project[1, :], ego_project[2, :]

# ╔═╡ bb65c59a-5f29-4453-a5a5-dae5856347c0
length(Graphs.neighborhood(model.social_network, node_id, depth))

# ╔═╡ 52dc63c8-c864-4075-91d8-8c0a98a6a717
Graphs.induced_subgraph(model.social_network, Graphs.neighborhood(model.social_network, node_id, depth))

# ╔═╡ c207e0c2-8713-4d88-95e0-4de41ef39c36
voterss = voters(model)

# ╔═╡ a768d815-2584-46a8-9328-be6cc8c02c92
get_votes(voterss)

# ╔═╡ 4079d722-1201-4b10-a2a2-9aa420089d67
social_network(model)

# ╔═╡ 09540198-bead-4df1-99bd-6e7848e7d215
md"### Play with diffusion"

# ╔═╡ 6f40b5c4-1252-472c-8932-11a2ee0935d2
md"Setup diffusion parameters and then check execution barrier for confirmation."

# ╔═╡ 0295587e-ad65-4bf1-b6d4-1b87a1e844ff
voter_diff_config_sp = Spearman_voter_diff_config(
			attract_proba = 0.8,
			change_rate = 0.5,
			normalize_shifts = (true, model_config.voter_config.weight_func(length(candidates)), model_config.voter_config.weight_func(1))
        )

# ╔═╡ 63a4233d-c596-4844-8539-91a2223f2266
voter_diff_config_kt = Kendall_voter_diff_config(attract_proba = 0.8)

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diffusion_config = Diffusion_config(
        checkpoint = 1,
		evolve_vertices = 0.1,
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
diffusions = 1

# ╔═╡ 27a60724-5d19-419f-b208-ffa0c78e2505
ensemble_size = 1

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

# ╔═╡ 0932e410-4a74-42b3-86c5-ac40f6be3543
md"##### Visualization of voters in vector space defined by their opinions"

# ╔═╡ 563c5195-fd67-48d8-9b01-a73ea756a7ba
md"Sampling:"

# ╔═╡ 228f2d0b-b964-4133-b0bc-fee7d9fe298f
begin
	sample_size = length(model.voters) < 1000 ? length(model.voters) :  ceil(Int, length(model.voters) * 0.01)
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)
end

# ╔═╡ 9f7f56db-8a73-40f7-bb47-9570e41a634f
md"Load voters from model:"

# ╔═╡ b203aed5-3231-4e19-a961-089c0a4cf8c6
md"Dimensionality reduction for visualisation of high dimensional opinions"

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
	_sampled_opinions = get_opinion(_model_log.voters[sampled_voter_ids])
	_projections = reduce_dim(_sampled_opinions, reduce_dim_config)
	
	x_projections = _projections[1, 1:9]
	y_projections = _projections[2, 1:9]
	
	min_x, max_x = minimum(_projections[1, :]), maximum(_projections[1, :]) 
	min_y, max_y = minimum(_projections[2, :]), maximum(_projections[2, :])
end

# ╔═╡ 52893c8c-d1b5-482a-aae7-b3ec5c590b77
md"Clustering for colouring of voters based on their opinions"

# ╔═╡ 1adc5d59-6198-4ef5-9a8a-6390acc28be1
clustering_config = Clustering_config(
    used = true,
    method = "K-means",
    kmeans_config = Kmeans_config(
        cluster_count = 8
    ),
    gm_config = GM_config(
    	cluster_count = 8
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
	sampled_voters = model_log.voters[sampled_voter_ids]
	sampled_opinions = get_opinion(sampled_voters)
end

# ╔═╡ 008cbbe1-949d-4be7-9178-9bfa27c6e2a8
begin
	projections = reduce_dim(sampled_opinions, reduce_dim_config)
	#unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)
	projections
end

# ╔═╡ c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
labels, clusters = clustering(sampled_opinions, candidates, length(parties), clustering_config)

# ╔═╡ f0eed2be-799f-46bb-968b-b70462cc06fd
title = reduce_dim_config.method * "_" * clustering_config.method * "_" * string(length(sampled_voters))

# ╔═╡ 83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
if step > 0
	prev_sampled_opinions = get_opinion(load_log(logger.exp_dir, step-1).voters[sampled_voter_ids])
	difference = sampled_opinions - prev_sampled_opinions
	changes = vec(sum(abs.(difference), dims=1))
	println(sum(changes))
	draw_heat_vis(projections, changes, "Heat map")
end

# ╔═╡ 18c07cb1-3a91-405c-a626-f608e0d1ea9d
OpinionDiffusion.GraphPlot.gplot(egon[1], x, y, nodelabel=egon[2], nodesize=[Graphs.degree(model.social_network, node) for node in egon[2]]) #nodefillc=cluster_colors[labels[egon[2]]])

# ╔═╡ 4eb56ee4-04e9-40cd-90fc-1495215f2928
draw_voter_vis(projections, clusters, title)

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
- Diameter"

# ╔═╡ 93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
function init_metrics(model, can_count)
	metrics = Dict()
	histogram = Graphs.degree_histogram(model.social_network)
    keyss = collect(keys(histogram))
	
	metrics["min_degrees"] = [minimum(keyss)]
	metrics["avg_degrees"] = [Graphs.ne(model.social_network) * 2 / Graphs.Graphs.nv(model.social_network)]
    metrics["max_degrees"] = [maximum(keyss)]
 
    #election results
	votes = get_votes(model.voters)
    metrics["plurality_votings"] = [plurality_voting(votes, can_count, true)]
    metrics["borda_votings"] = [borda_voting(votes, can_count, true)]
    metrics["copeland_votings"] = [copeland_voting(votes, can_count)]
	
	return metrics
end

# ╔═╡ 109e1b7c-16f0-4450-87ea-2d0442df5589
metrics = init_metrics(model, length(candidates))

# ╔═╡ 8d259bcc-d54c-401f-b864-87701f2bcf46
function update_metrics!(model, diffusion_metrics, can_count)
    dict = Graphs.degree_histogram(model.social_network)
    keyss = collect(keys(dict))
	
    push!(diffusion_metrics["min_degrees"], minimum(keyss))
    push!(diffusion_metrics["avg_degrees"], Graphs.ne(model.social_network) * 2 / Graphs.nv(model.social_network))
    push!(diffusion_metrics["max_degrees"], maximum(keyss))
    
    votes = get_votes(model.voters)
	
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
end

# ╔═╡ 805b55ad-c245-4140-84ce-c86e3b73a33c
if cb_run && ensemble_size > 1
	models, loggers = ensemble_model(logger.model_dir, ensemble_size)
	ensemble_metrics = [init_metrics(model, can_count) for model in models]
	
	for i in 1:diffusions
		run_ensemble!(models, diffusion_config, loggers)
		for (model, ensemble_metric) in zip(models, ensemble_metrics)
			update_metrics!(model, ensemble_metric, can_count)
		end
	end
end

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
# ╠═75588cfd-52c9-4406-975c-c03158db6e78
# ╟─46e8650e-f57d-48d7-89de-1c72e12dea45
# ╠═fc5a5935-8f4e-47ad-8568-70cd61656e06
# ╠═c6ccf2a8-e045-4da9-bbdb-270327c2d53f
# ╠═21f341c3-4d38-4449-b0f8-62e4ab14c99b
# ╠═042ec90d-8c0b-4acb-8d4c-31f701876cb6
# ╟─06d301e2-90c1-4fdf-b5ab-e127f8dee777
# ╠═d5d394c0-9b7c-4255-95fd-dc8cc32ca018
# ╠═8c709e84-66f8-4128-a85c-66a4c5ffc9b7
# ╠═20078431-5a7c-4a6f-b181-984ed54cf506
# ╠═70642f15-ac19-4a9c-993c-b567d336b43b
# ╠═28f103a9-2b18-4cfc-bcf3-34d512f8da03
# ╠═c50be339-8d1c-459f-8c14-934297f10717
# ╠═150ecd7a-5fe5-4e25-9630-91d26c30ff38
# ╠═089c39a6-e516-4aac-8014-b3a1e6444e61
# ╟─20893df2-fa42-46bf-889e-582b9ac39164
# ╠═519c9413-04ff-4a1e-995e-eaacf787df11
# ╠═affcdfab-a84b-4bc1-860d-17b05ac18133
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
# ╠═c207e0c2-8713-4d88-95e0-4de41ef39c36
# ╠═a768d815-2584-46a8-9328-be6cc8c02c92
# ╠═4079d722-1201-4b10-a2a2-9aa420089d67
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
# ╠═6492a611-fe57-4279-8852-9271b91396cc
# ╠═260af73d-28de-46da-8f80-54f4349e6fba
# ╠═805b55ad-c245-4140-84ce-c86e3b73a33c
# ╟─aad0da61-59d2-4429-97fa-6612872bb863
# ╟─55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
# ╠═ff873978-d93f-4ba2-aadd-6cfd3b136e3d
# ╟─0932e410-4a74-42b3-86c5-ac40f6be3543
# ╟─563c5195-fd67-48d8-9b01-a73ea756a7ba
# ╠═228f2d0b-b964-4133-b0bc-fee7d9fe298f
# ╠═462930eb-f995-48f0-9564-0c3b3d3b437f
# ╟─9f7f56db-8a73-40f7-bb47-9570e41a634f
# ╠═05eedffd-c82a-4dbd-8608-5fa00f9d0bae
# ╟─b203aed5-3231-4e19-a961-089c0a4cf8c6
# ╠═b8f16963-914b-40e4-b13d-77cb6eb7b6db
# ╠═e80cfc91-fc51-4216-a74c-39038d77c9db
# ╠═008cbbe1-949d-4be7-9178-9bfa27c6e2a8
# ╟─52893c8c-d1b5-482a-aae7-b3ec5c590b77
# ╠═1adc5d59-6198-4ef5-9a8a-6390acc28be1
# ╠═c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
# ╟─f0eed2be-799f-46bb-968b-b70462cc06fd
# ╟─0c2345e5-c1b2-4af8-ad31-617ddc99257a
# ╠═86659fc0-af7e-4498-8388-3e79349e9eb4
# ╠═83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
# ╠═18c07cb1-3a91-405c-a626-f608e0d1ea9d
# ╠═4eb56ee4-04e9-40cd-90fc-1495215f2928
# ╟─cd8ffe9d-e990-43ed-ad77-28067d1536ed
# ╟─d6c3ae90-2890-488f-8d76-d668236c0dc9
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═7f138d72-419a-4642-b163-6ec58ce42d24
# ╠═88ffc9b6-327b-4c6f-827b-67aa7e175855
# ╠═d0193993-832d-412d-8a3a-3b804b0845c7
# ╠═ae6f392e-6920-4e88-8587-837f03a00a88
# ╟─4c140c0e-71c7-4567-b37e-286395a450a3
# ╟─f539cf71-34ae-4e22-a7ac-d259b55cb2d3
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═8d259bcc-d54c-401f-b864-87701f2bcf46
# ╠═f6126696-ab8d-4637-b815-dd89ae8fb171
