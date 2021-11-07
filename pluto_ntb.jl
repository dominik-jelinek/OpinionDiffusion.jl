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

# ╔═╡ 75588cfd-52c9-4406-975c-c03158db6e78
import Distributions, Distances, Graphs

# ╔═╡ fc5a5935-8f4e-47ad-8568-70cd61656e06
input_filename = "ED-00001-00000003.toc"
#input_filename = "madeUp"

# ╔═╡ 76c03bc8-72b9-4fae-9310-3eb61d593896
@time parties, candidates, election = parse_data2(input_filename)

# ╔═╡ 228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
model_config = General_model_config( 
	m = 32,
	voter_config = Spearman_voter_config(
		weight_func = dist -> (1/2)^dist, 
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)
)

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir = "logs/" * "model_2021-10-18_21-46-20"

# ╔═╡ d8b613c5-7276-466d-a54a-7670f0921b35
exp_dir = model_dir * "/" * "experiment_2021-10-18_22-06-26"

# ╔═╡ 571e7a33-20b7-4432-b553-c07b9081c68d
idx = -1

# ╔═╡ 72450aaf-c6c4-458e-9555-39c31345116b
md"""
Model source $(@bind model_source Select(["restart_model" => "Restart", "load_model" => "Load", "new_model" => "New (takes time)"]))
"""

# ╔═╡ 1937642e-7f63-4ffa-b01f-22208a716dac
md"""
Execution barrier $(@bind cb_model CheckBox())
"""

# ╔═╡ 4a2b607d-947d-47e9-b73f-93eab1fb07a5
if cb_model 
	if model_source == "new_model"
		model = General_model(election, length(candidates), model_config)
		logger = Logger(model)
	elseif model_source == "load_model"
		model = load_log(exp_dir, idx)
		logger = Logger(model, model_dir, exp_dir, idx)
	else #restart
		model = load_log(model_dir)
		logger = Logger(model, model_dir)
	end
end

# ╔═╡ 776f99ea-6e44-4eb6-b2dd-3552bbb39954
logger.model_dir, logger.exp_dir

# ╔═╡ 4079d722-1201-4b10-a2a2-9aa420089d67
model.social_network

# ╔═╡ 26be9903-64f3-487f-af3f-dd1fc26c3665
Base.summarysize(model)

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diffusion_config = Diffusion_config(
        checkpoint = 1,
        voter_diff_config = Spearman_voter_diff_config(
            evolve_vertices = 0.1,
			attract_proba = 0.8,
			change_rate = 0.5,
            method = "averageAll"
        ),
        graph_diff_config = General_graph_diff_config(
            evolve_edges = 0.1,
            dist_metric = Distances.Cityblock(),
            edge_diff_func = dist -> (1/2)^(dist+6.28)
        )
    )

# ╔═╡ 93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
begin
	metrics = Dict()
	histogram = Graphs.degree_histogram(model.social_network)
    keyss = collect(keys(histogram))
	
	metrics["min_degrees"] = [minimum(keyss)]
	metrics["avg_degrees"] = [Graphs.ne(model.social_network) * 2 / Graphs.Graphs.nv(model.social_network)]
    metrics["max_degrees"] = [maximum(keyss)]
 
    #election results
	votes = get_votes(model.voters)
	can_count = length(candidates)
    metrics["plurality_votings"] = [plurality_voting(votes, can_count, true)]
    metrics["borda_votings"] = [borda_voting(votes, can_count, true)]
    metrics["copeland_votings"] = [copeland_voting(votes, can_count)]
	metrics
end

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

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
Execution barrier $(@bind cb_run CheckBox())
"""

# ╔═╡ d877c5d0-89af-48b9-bcd0-c1602d58339f
diffusions = 5

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb_run
	for i in 1:diffusions
		run!(model, diffusion_config, logger)
		update_metrics!(model, metrics, length(candidates))
	end
end

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
OpinionDiffusion.Plots.plotly()

# ╔═╡ 228f2d0b-b964-4133-b0bc-fee7d9fe298f
sample_size = ceil(Int, length(model.voters) * 0.05)

# ╔═╡ ea2d9183-07d1-454b-8e0a-1715528dc13e
sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)

# ╔═╡ 86659fc0-af7e-4498-8388-3e79349e9eb4
md"""Diffusion: $(@bind step Slider(0 : logger.diff_counter[1], show_value=true))"""

# ╔═╡ 462930eb-f995-48f0-9564-0c3b3d3b437f
model_log = load_log(logger.exp_dir, step)

# ╔═╡ 05eedffd-c82a-4dbd-8608-5fa00f9d0bae
sampled_voters = model_log.voters[sampled_voter_ids]

# ╔═╡ 7c2ebaff-c9ff-41f0-9825-28ca2d7d3d44
sampled_opinions = get_opinions(sampled_voters)

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
	
	x_projections = _projections[1, 1:10]
	y_projections = _projections[2, 1:10]
	
	min_x, max_x = minimum(_projections[1, :]), maximum(_projections[1, :]) 
	min_y, max_y = minimum(_projections[2, :]), maximum(_projections[2, :])
end

# ╔═╡ 008cbbe1-949d-4be7-9178-9bfa27c6e2a8
begin
	projections = reduce_dim(sampled_opinions, reduce_dim_config)
	
	unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)
	projections
end

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

# ╔═╡ c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
labels, clusters = clustering(sampled_opinions, candidates, parties, clustering_config)

# ╔═╡ f0eed2be-799f-46bb-968b-b70462cc06fd
title = reduce_dim_config.method * "_" * clustering_config.method * "_" * string(length(sampled_voters))

# ╔═╡ 7f138d72-419a-4642-b163-6ec58ce42d24
metrics_vis(metrics, candidates, parties)

# ╔═╡ 4eb56ee4-04e9-40cd-90fc-1495215f2928
draw_voter_vis(projections, clusters, title)

# ╔═╡ cd8ffe9d-e990-43ed-ad77-28067d1536ed
draw_degree_distr(Graphs.degree_histogram(model_log.social_network))

# ╔═╡ d6c3ae90-2890-488f-8d76-d668236c0dc9
draw_edge_distances(get_edge_distances(model_log.social_network, model_log.voters))

# ╔═╡ Cell order:
# ╠═86c32615-cdba-41aa-bfca-e5b90563f7f7
# ╠═481f2e27-0f88-482a-846a-6a31bf38f3ba
# ╠═9284976e-d474-11eb-2b94-dbe906a08bd7
# ╠═75588cfd-52c9-4406-975c-c03158db6e78
# ╠═fc5a5935-8f4e-47ad-8568-70cd61656e06
# ╠═76c03bc8-72b9-4fae-9310-3eb61d593896
# ╠═228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
# ╠═776f99ea-6e44-4eb6-b2dd-3552bbb39954
# ╠═985131a7-7c11-4f9d-ae00-ef031002592d
# ╠═d8b613c5-7276-466d-a54a-7670f0921b35
# ╠═571e7a33-20b7-4432-b553-c07b9081c68d
# ╟─72450aaf-c6c4-458e-9555-39c31345116b
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╠═4a2b607d-947d-47e9-b73f-93eab1fb07a5
# ╠═4079d722-1201-4b10-a2a2-9aa420089d67
# ╠═26be9903-64f3-487f-af3f-dd1fc26c3665
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═8d259bcc-d54c-401f-b864-87701f2bcf46
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═d877c5d0-89af-48b9-bcd0-c1602d58339f
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═ff873978-d93f-4ba2-aadd-6cfd3b136e3d
# ╠═228f2d0b-b964-4133-b0bc-fee7d9fe298f
# ╠═ea2d9183-07d1-454b-8e0a-1715528dc13e
# ╠═462930eb-f995-48f0-9564-0c3b3d3b437f
# ╠═86659fc0-af7e-4498-8388-3e79349e9eb4
# ╠═05eedffd-c82a-4dbd-8608-5fa00f9d0bae
# ╠═7c2ebaff-c9ff-41f0-9825-28ca2d7d3d44
# ╠═b8f16963-914b-40e4-b13d-77cb6eb7b6db
# ╠═e80cfc91-fc51-4216-a74c-39038d77c9db
# ╠═008cbbe1-949d-4be7-9178-9bfa27c6e2a8
# ╠═1adc5d59-6198-4ef5-9a8a-6390acc28be1
# ╠═c81ebbcb-c709-4713-a2b4-cf4bb48eb0da
# ╠═f0eed2be-799f-46bb-968b-b70462cc06fd
# ╠═7f138d72-419a-4642-b163-6ec58ce42d24
# ╟─4eb56ee4-04e9-40cd-90fc-1495215f2928
# ╠═cd8ffe9d-e990-43ed-ad77-28067d1536ed
# ╠═d6c3ae90-2890-488f-8d76-d668236c0dc9
