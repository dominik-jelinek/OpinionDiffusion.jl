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
import Distributions, Distances, Graphs

# ╔═╡ 20893df2-fa42-46bf-889e-582b9ac39164
md"### Setup model"

# ╔═╡ 228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
model_config = General_model_config( 
	m = 32,
	voter_config = Spearman_voter_config(
		weight_func = position -> (1/2)^position, 
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)
)

# ╔═╡ fc5a5935-8f4e-47ad-8568-70cd61656e06
input_filename = "ED-00001-00000003.toc"
#input_filename = "madeUp.toc"

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir = "logs/" * "model_2021-11-07_20-35-37"

# ╔═╡ d8b613c5-7276-466d-a54a-7670f0921b35
exp_dir = model_dir * "/" * "experiment_2021-11-07_20-35-37"

# ╔═╡ 4712586c-bc93-43df-ae66-1e75a21b6f85
md"Index for loading specific model state inside of exp_dir. Insert -1 for the last state of the model."

# ╔═╡ 571e7a33-20b7-4432-b553-c07b9081c68d
idx = -1

# ╔═╡ e9169286-8d05-4bc9-8dc4-16ae6bd81038
logging = true

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

# ╔═╡ c6ccf2a8-e045-4da9-bbdb-270327c2d53f
begin
	parties, candidates, election = parse_data(input_filename)
	can_count = length(candidates)
end

# ╔═╡ 93cb669f-6743-4b50-80de-c8594ea20497
if cb_model && logging
	if model_source == "new_model"
		model = General_model(election, can_count, model_config)
		logger = Logger(model)
	elseif model_source == "load_model"
		model, logger = load_model(model_dir, exp_dir, idx, true)
	else # restart
		model, logger = restart_model(model_dir)
	end
elseif cb_model && !logging
	if model_source == "new_model"
		model = General_model(election, can_count, model_config)
	elseif model_source == "load_model"
		model = load_log(exp_dir, idx)
	else # restart
		model = load_log(model_dir)
	end
end

# ╔═╡ 776f99ea-6e44-4eb6-b2dd-3552bbb39954
logger.model_dir, logger.exp_dir

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

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diffusion_config = Diffusion_config(
        checkpoint = 1,
        voter_diff_config = Spearman_voter_diff_config(
            evolve_vertices = 0.1,
			attract_proba = 0.8,
			change_rate = 0.5,
			normalize_shifts = (true, model_config.voter_config.weight_func(can_count), model_config.voter_config.weight_func(1))
        ),
        graph_diff_config = General_graph_diff_config(
            evolve_edges = 0.1,
            dist_metric = Distances.Cityblock(),
            edge_diff_func = dist -> (1/2)^(dist+6.28)
        )
    )

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
Execution barrier $(@bind cb_run CheckBox())
"""

# ╔═╡ d877c5d0-89af-48b9-bcd0-c1602d58339f
diffusions = 10

# ╔═╡ 27a60724-5d19-419f-b208-ffa0c78e2505
ensemble_size = 1

# ╔═╡ aad0da61-59d2-4429-97fa-6612872bb863
md"### Diffusion analyzation"

# ╔═╡ 55836ee2-8ea9-4b42-bdcc-3f5dab0cef20
md"Choose visualization backend from Plots library:"

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
OpinionDiffusion.Plots.plotly()

# ╔═╡ 0932e410-4a74-42b3-86c5-ac40f6be3543
md"##### Visualization of voters in vector space defined by their opinions"

# ╔═╡ 563c5195-fd67-48d8-9b01-a73ea756a7ba
md"Sampling:"

# ╔═╡ 228f2d0b-b964-4133-b0bc-fee7d9fe298f
begin
	sample_size = length(model.voters) < 1000 ? length(model.voters) :  ceil(Int, length(model.voters) * 0.05)
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
	_sampled_opinions = get_opinions(_model_log.voters[sampled_voter_ids])
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
    method = "Party",
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
md"""Diffusion: $(@bind step Slider(0 : logger.diff_counter[1], show_value=true))"""

# ╔═╡ 462930eb-f995-48f0-9564-0c3b3d3b437f
model_log = load_log(logger.exp_dir, step)

# ╔═╡ 05eedffd-c82a-4dbd-8608-5fa00f9d0bae
begin
	sampled_voters = model_log.voters[sampled_voter_ids]
	sampled_opinions = get_opinions(sampled_voters)
end

# ╔═╡ 008cbbe1-949d-4be7-9178-9bfa27c6e2a8
begin
	projections = reduce_dim(sampled_opinions, reduce_dim_config)
	
	unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)
	projections
end

# ╔═╡ c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
labels, clusters = clustering(sampled_opinions, candidates, parties, clustering_config)

# ╔═╡ f0eed2be-799f-46bb-968b-b70462cc06fd
title = reduce_dim_config.method * "_" * clustering_config.method * "_" * string(length(sampled_voters))

# ╔═╡ 83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
if step > 0
	prev_sampled_opinions = get_opinions(load_log(logger.exp_dir, step-1).voters[sampled_voter_ids])
	difference = sampled_opinions - prev_sampled_opinions
	draw_heat_vis(projections, vec(sum(abs.(difference), dims=1)), "Heat map")
end

# ╔═╡ 4eb56ee4-04e9-40cd-90fc-1495215f2928
draw_voter_vis(projections, clusters, title)

# ╔═╡ cd8ffe9d-e990-43ed-ad77-28067d1536ed
draw_degree_distr(Graphs.degree_histogram(model_log.social_network))

# ╔═╡ d6c3ae90-2890-488f-8d76-d668236c0dc9
draw_edge_distances(get_edge_distances(model_log.social_network, model_log.voters))

# ╔═╡ d716423e-7945-4e0a-a6ab-17e0b94c721e
md"##### Compounded metrics for all diffusions"

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
metrics = init_metrics(model, can_count)

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
		update_metrics!(model, metrics, can_count)
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

    plots = OpinionDiffusion.Plots.plot(degrees, plurality, borda, copeland, layout = (2, 2), size = (980,1200))
    
    if exp_dir != Nothing
        OpinionDiffusion.Plots.savefig(plots, "$(exp_dir)/images/metrics.png")
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
# ╟─20893df2-fa42-46bf-889e-582b9ac39164
# ╠═228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
# ╠═fc5a5935-8f4e-47ad-8568-70cd61656e06
# ╠═776f99ea-6e44-4eb6-b2dd-3552bbb39954
# ╠═985131a7-7c11-4f9d-ae00-ef031002592d
# ╠═d8b613c5-7276-466d-a54a-7670f0921b35
# ╟─4712586c-bc93-43df-ae66-1e75a21b6f85
# ╠═571e7a33-20b7-4432-b553-c07b9081c68d
# ╠═e9169286-8d05-4bc9-8dc4-16ae6bd81038
# ╟─98885ec6-7561-43d7-bdf6-7f58fb2720f6
# ╟─72450aaf-c6c4-458e-9555-39c31345116b
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╠═c6ccf2a8-e045-4da9-bbdb-270327c2d53f
# ╠═93cb669f-6743-4b50-80de-c8594ea20497
# ╠═c207e0c2-8713-4d88-95e0-4de41ef39c36
# ╠═a768d815-2584-46a8-9328-be6cc8c02c92
# ╠═4079d722-1201-4b10-a2a2-9aa420089d67
# ╟─09540198-bead-4df1-99bd-6e7848e7d215
# ╟─6f40b5c4-1252-472c-8932-11a2ee0935d2
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═d877c5d0-89af-48b9-bcd0-c1602d58339f
# ╠═27a60724-5d19-419f-b208-ffa0c78e2505
# ╠═109e1b7c-16f0-4450-87ea-2d0442df5589
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
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
# ╟─e80cfc91-fc51-4216-a74c-39038d77c9db
# ╟─008cbbe1-949d-4be7-9178-9bfa27c6e2a8
# ╟─52893c8c-d1b5-482a-aae7-b3ec5c590b77
# ╠═1adc5d59-6198-4ef5-9a8a-6390acc28be1
# ╠═c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
# ╟─f0eed2be-799f-46bb-968b-b70462cc06fd
# ╟─0c2345e5-c1b2-4af8-ad31-617ddc99257a
# ╠═86659fc0-af7e-4498-8388-3e79349e9eb4
# ╠═83c88ccd-793a-42f5-b2b8-ae9dbf5e702d
# ╟─4eb56ee4-04e9-40cd-90fc-1495215f2928
# ╟─cd8ffe9d-e990-43ed-ad77-28067d1536ed
# ╟─d6c3ae90-2890-488f-8d76-d668236c0dc9
# ╟─d716423e-7945-4e0a-a6ab-17e0b94c721e
# ╠═7f138d72-419a-4642-b163-6ec58ce42d24
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═8d259bcc-d54c-401f-b864-87701f2bcf46
# ╠═f6126696-ab8d-4637-b815-dd89ae8fb171
