### A Pluto.jl notebook ###
# v0.19.19

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

# ╔═╡ 521fad35-0852-48a1-93b0-7b8794544706
using Pkg

# ╔═╡ f1bbf513-b421-492d-80b9-fe6b9c8373c1
Pkg.activate()

# ╔═╡ 68da078f-4b9a-4254-9c71-6a6c31761963
using Revise

# ╔═╡ b6b67170-8dcf-11ed-22a8-c935c756f9b0
using Profile

# ╔═╡ 8d2cc22a-72fa-4657-a0a2-6b1d586ec465
using OpinionDiffusion, PlutoUI

# ╔═╡ 8719fef4-915d-420f-9434-22a20e6d790d
using KernelDensity

# ╔═╡ 6076b5d6-9fc0-47ba-8d90-059f80d941c9
using GraphMakie, CairoMakie

# ╔═╡ 6ff48f11-656b-4084-acc7-fc7c4dba7d7c
import Distributions, Distances, Graphs, Plots, Random

# ╔═╡ 3d2caf1f-c781-40be-944f-f85b3f56a42a
Plots.plotly()
#Plots.gr()

# ╔═╡ e12063e2-9afc-472a-b0d0-eb4eee4d9fb8
md"## Configure visualizations"

# ╔═╡ 8aaf35b2-3c2c-45f9-afc0-2fe472f7cbbc
log_dir_path = "./logs"

# ╔═╡ 02d8b06f-0add-4b9d-95da-25683e4ded82
md"""Select model: $(@bind model_dir Select([filename for filename in readdir( log_dir_path) if split(filename, "_")[1] == "model"]))"""

# ╔═╡ 8d05a1b3-2501-47f2-abed-6362e1a6c946
model_dir_path = log_dir_path * "/" * model_dir

# ╔═╡ 62535f41-dccc-46b9-9393-a833d6d1831b
md"""Select experiment: $(@bind exp_dir Select([filename for filename in readdir(model_dir_path) if split(filename, "_")[1] == "experiment"]))"""

# ╔═╡ 531136e8-2b27-4b76-aa4b-c84a9bcbe567
exp_dir_path = model_dir_path * "/" * exp_dir

# ╔═╡ 27ce1f4d-4cb1-4fac-85b9-fdcb71cfb8f1
md"""Select diffusion step: $(@bind step Select([parse(Int64, split(file, "_")[2][1:end-5]) for file in readdir(exp_dir_path)]))"""

# ╔═╡ c2ba23c6-d23b-4754-a5b8-371500420b43
model_log = load_log(exp_dir_path, step)

# ╔═╡ daccad47-5400-4a1b-b73d-f920e2bdc78a
candidates = get_candidates(model_log)

# ╔═╡ f00d2476-b96a-4595-a812-20cace428b75
begin
	sample_size = min(1024, length(model_log.voters))
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model_log.voters), sample_size, replace=false)
end

# ╔═╡ 31b2fe2a-a778-4893-9188-c38d6e4b21e2
sampled_voters = get_voters(model_log)[sampled_voter_ids]

# ╔═╡ e1599782-86c1-443e-8f4f-59e47fdd5f86
md"""Dimensionality reduction method: $(@bind dim_reduction_method Select(["PCA", "Tsne", "MDS"], default="PCA"))"""

# ╔═╡ 80a5d0d1-6522-4464-9eb4-bfd46b2124a1
out_dim = 2

# ╔═╡ 2b59335e-14d5-4fee-acc7-af1e5e458c09
if dim_reduction_method == "PCA"
	dim_reduction_config = PCA_dim_reduction_config(out_dim)
elseif dim_reduction_method == "Tsne"
	dim_reduction_config = Tsne_dim_reduction_config(out_dim = out_dim,
        											reduce_dims = 0,
        											max_iter = 3000,
        											perplexity = 100.0)
else dim_reduction_method == "MDS"
	dim_reduction_config = MDS_dim_reduction_config(out_dim)
end

# ╔═╡ de79e133-798a-490b-a4c9-9f38d5f31a98
md"""Clustering method: $(@bind clustering_method Select(["Kmeans", "GM", "Party", "DBSCAN", "Density"], default="Density"))"""

# ╔═╡ 5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
title = dim_reduction_method * "_" * clustering_method * "_" * string(length(sampled_voters))

# ╔═╡ 9bf7d31b-7a4e-4ec9-99b9-d189c9b2297e
begin
	projections = reduce_dims(sampled_voters, dim_reduction_config)
	#unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)
	projections
end

# ╔═╡ 60d1816d-5dd3-4166-bb11-8cd2307bc9ee
if clustering_method == "Party"
	clustering_config = Party_clustering_config(candidates)
elseif clustering_method == "Kmeans"
	clustering_config = Kmeans_clustering_config(length(candidates))
elseif clustering_method == "GM"
	clustering_config = GM_clustering_config(length(candidates))
elseif clustering_method == "DBSCAN"
	clustering_config = DBSCAN_clustering_config(0.1, 3)
else
	clustering_config = Density_clustering_config(3, projections)
end

# ╔═╡ 1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
labels, clusters, plt = clustering(sampled_voters, clustering_config)

# ╔═╡ 5000e501-4453-496c-90b9-ff312223b4d4
plt

# ╔═╡ ad8f309f-b3f1-4ed0-914f-675ffe23821f
visualizations = OpinionDiffusion.gather_vis2(exp_dir_path, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ f18be3eb-2007-46fe-9b9a-782796068f7f
visualizations[1][3]

# ╔═╡ d62e9041-1b7a-4abe-8097-20d98fd3659a
if step > 0
	prev_sampled_opinions = reduce(hcat, get_opinion(load_log(logger.exp_dir, step-1).voters[sampled_voter_ids]))
	difference = sampled_opinions - prev_sampled_opinions
	changes = vec(sum(abs.(difference), dims=1))
	println(sum(changes))
	draw_heat_vis(projections, changes, "Heat map")
end

# ╔═╡ 6f79058e-2243-4be1-aaf6-7e95d6f7c342
begin
	_model_log = load_log(logger.exp_dir, 0)
	_projections = reduce_dims(sampled_voters, dim_reduction_config)
	
	x_projections = _projections[1, 1:length(candidates)]
	y_projections = _projections[2, 1:length(candidates)]
	
	min_x, max_x = minimum(_projections[1, :]), maximum(_projections[1, :]) 
	min_y, max_y = minimum(_projections[2, :]), maximum(_projections[2, :])
end

# ╔═╡ 127c5ea4-ee23-4e40-81f7-558582c130c6
dens = kde((projections[1, :], projections[2, :]))

# ╔═╡ 58c0d57c-8cad-4150-8c5e-88cb3152d2e5
density = round.(dens.density; digits=4) 

# ╔═╡ 603e34db-fffe-4e25-ac72-618d060912cf
Plots.heatmap(density, title="Voter map", c=Plots.cgrad([:blue, :white,:red, :yellow]))

# ╔═╡ 5fbe700d-1817-4f2c-87f8-86c6a812cbdf
maximum(density) / 100

# ╔═╡ 1bb69cb2-a181-4ef7-b978-4374d8e3c813
visualizations[diff][2]

# ╔═╡ 5bb6293b-6dba-4843-ad08-373a0003b46e
function find_local_maxima(a)
    max_indices = Vector{Int64}()

    start = 1
    peak = a[1] == a[2] ? true : false
    if a[1] > a[2]
        push!(max_indices, 1)
    end

    for i in 2:length(a) - 1
        # test inflection point
        if peak && a[i] < a[i + 1]
            peak = false
        end

        # test start of the peak
        if a[i - 1] < a[i] && a[i] >= a[i + 1]
            start = i
            peak = true
        end

        # test end of the peak
        if peak && a[i] > a[i + 1]
            append!(max_indices, collect(start:i))
            peak = false
        end
    end

    if peak && a[end - 1] == a[end]
        append!(max_indices, collect(start:length(a)))
    end

    if a[end - 1] < a[end]
        push!(max_indices, length(a))
    end

    return max_indices
end

# ╔═╡ e9066379-05df-4b01-a498-da0e614d3baa
begin
	peaks_mask_y = zeros(Int64, size(density, 1), size(density, 2))
	for (i, col) in enumerate(eachcol(density)) peaks_mask_y[find_local_maxima(col), i] .= 1 end 
end

# ╔═╡ e5a064f1-4715-4c6b-b331-864b0b8503b8
begin
	drops_mask_y = zeros(Int64, size(density, 1), size(density, 2))
	for (i, col) in enumerate(eachcol(density)) drops_mask_y[find_local_maxima(col .* -1), i] .= -1 end 
end

# ╔═╡ a4326151-238f-48f7-beb3-d0bab2aa250f
begin
	peaks_mask_x = zeros(Int64, size(density, 1), size(density, 2))
	for (i, row) in enumerate(eachrow(density)) peaks_mask_x[i, find_local_maxima(row)] .= 1 end 
end

# ╔═╡ 29ebcf06-6e43-41fc-a403-17da32ad31c2
Plots.heatmap(peaks_mask_y .& peaks_mask_x)

# ╔═╡ 0cdc3faf-151b-45d6-8fe4-63d7a0cea57b
begin
	drops_mask_x = zeros(Int64, size(density, 1), size(density, 2))
	for (i, row) in enumerate(eachrow(density)) drops_mask_x[i, find_local_maxima(row .* -1)] .= -1 end 
end

# ╔═╡ ec7e4011-fe74-434a-a5b1-22dfec514d49
Plots.heatmap((drops_mask_y .| drops_mask_x) .| (peaks_mask_y .& peaks_mask_x))

# ╔═╡ 08948c84-c8d1-4b4d-87a6-113fdd3596f6
heatmap(dens, colormap=[:blue, :white,:red, :yellow])

# ╔═╡ 05f14a3d-80a7-4ea6-8466-6f4d03fbe173
begin 
	social_network = get_social_network(model)
    voters = get_voters(model)
	
    cluster_graph = OpinionDiffusion.get_cluster_graph(model, clusters, labels, projections)
	
    println(Graphs.modularity(cluster_graph, labels))
end

# ╔═╡ ed8fe954-4a93-4e89-9667-ef73b8e3495e
cluster_metrics = OpinionDiffusion.cluster_graph_metrics(cluster_graph, social_network, voters, length(candidates))

# ╔═╡ a35ca262-f5cc-404f-8c39-5d73e3477321
OpinionDiffusion.draw_cluster_graph(cluster_graph, cluster_metrics)

# ╔═╡ 00bf833f-22d5-47c3-ba59-6d7841bd93f0
md"### Metrics specific for every timestamp"

# ╔═╡ 7959b6c6-e501-4bd1-943d-fd8fd0efe4be
@bind clk Clock()

# ╔═╡ 902b2b49-5575-4b9f-aa7b-88583e213314
t = clk % length(visualizations) + 1

# ╔═╡ 7d2c5850-0ae9-44ea-ac31-130bc248e2ce
visualizations[t]

# ╔═╡ eacbf9e0-8191-46fc-ab5f-186a244a7aae
md"### Node visualization"

# ╔═╡ 846390ef-0817-48c6-889a-18fedf2a8df3
node_id = 42

# ╔═╡ 867261f8-89b6-42cb-a616-a4f3c553d522
depth = 1

# ╔═╡ bff33fa7-3bd4-4534-acab-1ce2fe2b57c2
Graphs.induced_subgraph(model.social_network, Graphs.neighborhood(model.social_network, node_id, depth))

# ╔═╡ af31098c-bba9-42f2-8a72-e03f51f26ff3
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

# ╔═╡ 1d09cd30-6119-4718-a4cc-385f57862fcc
egon = ego(model.social_network, 100, depth)

# ╔═╡ c2fa22d9-51ff-4db1-bd88-32c1f494473f
typeof(egon[1])

# ╔═╡ 64945f43-c9e7-4209-ad8a-4c81f0ac9b5a
ego_project = reduce_dim(sampled_opinions, reduce_dim_config)[:, egon[2]]

# ╔═╡ fbc0593b-917a-44e2-8e1f-691839851e5c
x, y = ego_project[1, :], ego_project[2, :]

# ╔═╡ 94c32e1c-8c12-4b49-95b6-ad7dfdf2c827
OpinionDiffusion.GraphPlot.gplot(egon[1], x, y, nodesize=[Graphs.degree(model.social_network, node) for node in egon[2]]) #nodefillc=cluster_colors[labels[egon[2]]])

# ╔═╡ fc865475-463f-40b1-9481-35123adcbffa
[sortperm(borda_voting(get_votes(sampled_voters[collect(cluster)]), length(candidates), true), rev=true) for cluster in clusters if length(cluster) != 0]

# ╔═╡ 2750e431-7c7f-4323-b74b-399ab7346603
countss = [get_counts(get_votes(sampled_voters[collect(cluster)]), length(candidates)) for cluster in clusters if length(cluster) != 0]

# ╔═╡ fc58665b-955b-4091-bd50-17ee0f8539ae
plots = Plots.plot([Plots.heatmap(count, yticks=1:length(src_candidates), xticks=1:length(src_candidates), ylabel="Candidate", xlabel="Position") for count in countss]..., layout = (length(candidates), 1), size = (669,900))

# ╔═╡ Cell order:
# ╠═b6b67170-8dcf-11ed-22a8-c935c756f9b0
# ╠═521fad35-0852-48a1-93b0-7b8794544706
# ╠═f1bbf513-b421-492d-80b9-fe6b9c8373c1
# ╠═68da078f-4b9a-4254-9c71-6a6c31761963
# ╠═8d2cc22a-72fa-4657-a0a2-6b1d586ec465
# ╠═8719fef4-915d-420f-9434-22a20e6d790d
# ╠═6ff48f11-656b-4084-acc7-fc7c4dba7d7c
# ╠═6076b5d6-9fc0-47ba-8d90-059f80d941c9
# ╠═3d2caf1f-c781-40be-944f-f85b3f56a42a
# ╟─e12063e2-9afc-472a-b0d0-eb4eee4d9fb8
# ╟─8aaf35b2-3c2c-45f9-afc0-2fe472f7cbbc
# ╟─02d8b06f-0add-4b9d-95da-25683e4ded82
# ╟─8d05a1b3-2501-47f2-abed-6362e1a6c946
# ╟─62535f41-dccc-46b9-9393-a833d6d1831b
# ╟─531136e8-2b27-4b76-aa4b-c84a9bcbe567
# ╟─27ce1f4d-4cb1-4fac-85b9-fdcb71cfb8f1
# ╠═c2ba23c6-d23b-4754-a5b8-371500420b43
# ╠═daccad47-5400-4a1b-b73d-f920e2bdc78a
# ╠═f00d2476-b96a-4595-a812-20cace428b75
# ╠═31b2fe2a-a778-4893-9188-c38d6e4b21e2
# ╟─e1599782-86c1-443e-8f4f-59e47fdd5f86
# ╠═80a5d0d1-6522-4464-9eb4-bfd46b2124a1
# ╠═2b59335e-14d5-4fee-acc7-af1e5e458c09
# ╠═de79e133-798a-490b-a4c9-9f38d5f31a98
# ╠═1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
# ╠═60d1816d-5dd3-4166-bb11-8cd2307bc9ee
# ╠═5000e501-4453-496c-90b9-ff312223b4d4
# ╠═ec7e4011-fe74-434a-a5b1-22dfec514d49
# ╠═603e34db-fffe-4e25-ac72-618d060912cf
# ╠═5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
# ╠═d62e9041-1b7a-4abe-8097-20d98fd3659a
# ╠═fc58665b-955b-4091-bd50-17ee0f8539ae
# ╠═ad8f309f-b3f1-4ed0-914f-675ffe23821f
# ╠═f18be3eb-2007-46fe-9b9a-782796068f7f
# ╠═9bf7d31b-7a4e-4ec9-99b9-d189c9b2297e
# ╠═6f79058e-2243-4be1-aaf6-7e95d6f7c342
# ╠═127c5ea4-ee23-4e40-81f7-558582c130c6
# ╠═58c0d57c-8cad-4150-8c5e-88cb3152d2e5
# ╠═5fbe700d-1817-4f2c-87f8-86c6a812cbdf
# ╠═e9066379-05df-4b01-a498-da0e614d3baa
# ╠═e5a064f1-4715-4c6b-b331-864b0b8503b8
# ╠═a4326151-238f-48f7-beb3-d0bab2aa250f
# ╠═0cdc3faf-151b-45d6-8fe4-63d7a0cea57b
# ╠═1bb69cb2-a181-4ef7-b978-4374d8e3c813
# ╠═29ebcf06-6e43-41fc-a403-17da32ad31c2
# ╠═5bb6293b-6dba-4843-ad08-373a0003b46e
# ╠═08948c84-c8d1-4b4d-87a6-113fdd3596f6
# ╠═05f14a3d-80a7-4ea6-8466-6f4d03fbe173
# ╠═ed8fe954-4a93-4e89-9667-ef73b8e3495e
# ╠═a35ca262-f5cc-404f-8c39-5d73e3477321
# ╠═00bf833f-22d5-47c3-ba59-6d7841bd93f0
# ╠═7959b6c6-e501-4bd1-943d-fd8fd0efe4be
# ╠═902b2b49-5575-4b9f-aa7b-88583e213314
# ╠═7d2c5850-0ae9-44ea-ac31-130bc248e2ce
# ╟─eacbf9e0-8191-46fc-ab5f-186a244a7aae
# ╠═846390ef-0817-48c6-889a-18fedf2a8df3
# ╠═867261f8-89b6-42cb-a616-a4f3c553d522
# ╠═1d09cd30-6119-4718-a4cc-385f57862fcc
# ╠═c2fa22d9-51ff-4db1-bd88-32c1f494473f
# ╠═64945f43-c9e7-4209-ad8a-4c81f0ac9b5a
# ╠═fbc0593b-917a-44e2-8e1f-691839851e5c
# ╠═94c32e1c-8c12-4b49-95b6-ad7dfdf2c827
# ╠═bff33fa7-3bd4-4534-acab-1ce2fe2b57c2
# ╠═af31098c-bba9-42f2-8a72-e03f51f26ff3
# ╠═fc865475-463f-40b1-9481-35123adcbffa
# ╠═2750e431-7c7f-4323-b74b-399ab7346603
