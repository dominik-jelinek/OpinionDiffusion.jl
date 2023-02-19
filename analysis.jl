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

# ╔═╡ 2dad0bf5-6d17-4bf5-92ca-c77421d0dba3
TableOfContents(depth=4)

# ╔═╡ e12063e2-9afc-472a-b0d0-eb4eee4d9fb8
md"## Configure visualizations"

# ╔═╡ 44bdb36d-cf1e-4a5a-b97c-5b5d9f45b54a
md"### Dimensionality reduction"

# ╔═╡ 80a5d0d1-6522-4464-9eb4-bfd46b2124a1
out_dim = 2

# ╔═╡ e1599782-86c1-443e-8f4f-59e47fdd5f86
md"""Dimensionality reduction method: $(@bind dim_reduction_method Select(["PCA", "Tsne", "MDS"], default="PCA"))"""

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

# ╔═╡ 6e9b9800-3ea8-432c-905f-de2049a6021a
md"### Clustering"

# ╔═╡ de79e133-798a-490b-a4c9-9f38d5f31a98
md"""Clustering method: $(@bind clustering_method Select(["Kmeans", "GM", "Party", "DBSCAN", "Density"], default="Density"))"""

# ╔═╡ 4d57266e-0800-4f95-ac49-3a2ba791bcb4
md"## Diffusion visualization"

# ╔═╡ 5d80608d-74e3-4d0a-a65d-4231447f4ba3
md"### Loading data"

# ╔═╡ 8aaf35b2-3c2c-45f9-afc0-2fe472f7cbbc
log_dir_path = "./logs"

# ╔═╡ 02d8b06f-0add-4b9d-95da-25683e4ded82
md"""Select model: $(@bind model_dir Select([filename for filename in readdir( log_dir_path) if split(filename, "_")[1] == "model" && split(filename, "_")[2] != "ensemble"]))"""

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

# ╔═╡ 60d1816d-5dd3-4166-bb11-8cd2307bc9ee
if clustering_method == "Party"
	clustering_config = Party_clustering_config(candidates)
elseif clustering_method == "Kmeans"
	clustering_config = Kmeans_clustering_config(length(candidates))
elseif clustering_method == "GM"
	clustering_config = GM_clustering_config(length(candidates)-3)
elseif clustering_method == "DBSCAN"
	clustering_config = DBSCAN_clustering_config(0.1, 3)
else
	clustering_config = Density_clustering_config(3)
end

# ╔═╡ 67d59347-2491-44f8-ad07-91f55e314d0e
md"### Sampling"

# ╔═╡ f00d2476-b96a-4595-a812-20cace428b75
begin
	sample_size = min(1024, length(model_log.voters))
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model_log.voters), sample_size, replace=false)
end

# ╔═╡ 31b2fe2a-a778-4893-9188-c38d6e4b21e2
sampled_voters = get_voters(model_log)[sampled_voter_ids]

# ╔═╡ 9bf7d31b-7a4e-4ec9-99b9-d189c9b2297e
begin
	projections = reduce_dims(sampled_voters, dim_reduction_config)
	#unify_projections!(projections, x_projections, y_projections, (max_x-min_x)/2, (max_y-min_y)/2)
	projections
end

# ╔═╡ 1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
labels, clusters = clustering(sampled_voters, clustering_config, projections)

# ╔═╡ c09828d7-82c0-40cf-ba0f-963ff1f7d8f7
clusters

# ╔═╡ 23c54f4b-6f03-4661-b9c2-5ace1028e577
dists = get_distance(sampled_voters)

# ╔═╡ 79f2fead-eff3-4a74-ac65-39dd5d51d361
labels[1]

# ╔═╡ df5bec5f-fbb2-4f81-9398-c1b1ba317d1b
for i in eachindex(clusters)
	println(clusters[i][1], ": ", OpinionDiffusion.Statistics.mean(dists[1, collect(clusters[i][2])]))
end

# ╔═╡ 2ef976be-ea40-446e-bbc0-3fb4e1c29dc7
function remove_holes(labels::Vector{Int64})
    unique_labels = sort(unique(labels)) # get the unique labels
	mapping = Dict(value => i for (i, value) in enumerate(unique_labels))
	return mapping
end

# ╔═╡ 8b5e91c6-3355-4e23-abd5-3f67d4689c1c
clusters

# ╔═╡ a1fc11c2-b0a9-422e-a2f0-cc28ff063c50
sort(unique(labels))

# ╔═╡ 03c6c2bf-364f-448e-969f-009fedaeb989
mapping = remove_holes(labels)

# ╔═╡ 195c6d1d-645e-4b7c-9d32-8f56359ebf99
new_labels = [mapping[label] for label in labels]

# ╔═╡ b425603d-e34d-4f95-a3d9-1bfce2982cfd
sils = OpinionDiffusion.Clustering.silhouettes(new_labels, dists)

# ╔═╡ 05c64c67-e51a-49cc-8908-9d035fe18664
OpinionDiffusion.Statistics.median(sils)

# ╔═╡ ee682d4f-705d-43a8-b379-7b74af63792d
heatmap(-density, colormap=[:blue, :white,:red, :yellow])

# ╔═╡ 9f49b73a-eea6-4533-971d-bafb322183e0
@bind diff Clock(interval=0.2, fixed=true, max_value=length(plt))

# ╔═╡ c245c7c0-fbe0-4e6a-8e6e-91fc997bd9df
 #md"Diff: $(@bind diff PlutoUI.Slider(1:100))"

# ╔═╡ a1d5177c-6540-475d-bae3-1cedfd692d7b
plt[diff]

# ╔═╡ 33ed15ce-e845-40d7-9423-4071f64f75d1


# ╔═╡ 5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
title = dim_reduction_method * "_" * clustering_method * "_" * string(length(sampled_voters))

# ╔═╡ d62e9041-1b7a-4abe-8097-20d98fd3659a
if step > 0
	prev_sampled_opinions = reduce(hcat, get_opinion(load_log(logger.exp_dir, step-1).voters[sampled_voter_ids]))
	difference = sampled_opinions - prev_sampled_opinions
	changes = vec(sum(abs.(difference), dims=1))
	println(sum(changes))
	draw_heat_vis(projections, changes, "Heat map")
end

# ╔═╡ ad8f309f-b3f1-4ed0-914f-675ffe23821f
begin
	visualizations = OpinionDiffusion.gather_vis2(exp_dir_path, sampled_voter_ids, dim_reduction_config, clustering_config)
	println("Done")
end

# ╔═╡ f18be3eb-2007-46fe-9b9a-782796068f7f
visualizations[1][3]

# ╔═╡ bca24178-9793-4ab2-b7ab-861f9093e652
Makie.plot!(fig[1], visualizations[1][3])

# ╔═╡ 5b954687-4cfc-4298-88f3-0b82e3877da8
typeof(visualizations[1][3])

# ╔═╡ 6f79058e-2243-4be1-aaf6-7e95d6f7c342
begin
	_model_log = load_log(logger.exp_dir, 0)
	_projections = reduce_dims(sampled_voters, dim_reduction_config)
	
	x_projections = _projections[1, 1:length(candidates)]
	y_projections = _projections[2, 1:length(candidates)]
	
	min_x, max_x = minimum(_projections[1, :]), maximum(_projections[1, :]) 
	min_y, max_y = minimum(_projections[2, :]), maximum(_projections[2, :])
end

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

# ╔═╡ 218a9121-2633-493c-b126-5ddc5dcd8767
md"### Compounded metrics"

# ╔═╡ 00bf833f-22d5-47c3-ba59-6d7841bd93f0
md"### Specific timestep visualizations"

# ╔═╡ 59cd2b6a-2f14-419f-be29-ef60b0f1c681
md"#### Communities"

# ╔═╡ 7959b6c6-e501-4bd1-943d-fd8fd0efe4be
@bind clk Clock()

# ╔═╡ 902b2b49-5575-4b9f-aa7b-88583e213314
t = clk % length(visualizations) + 1

# ╔═╡ 7d2c5850-0ae9-44ea-ac31-130bc248e2ce
visualizations[t][3]

# ╔═╡ b3a2f654-9e0e-4fcf-bf35-924e097427fe
visualizations[t][1]

# ╔═╡ eacbf9e0-8191-46fc-ab5f-186a244a7aae
md"### Specific node visualizations"

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
# ╠═2dad0bf5-6d17-4bf5-92ca-c77421d0dba3
# ╟─e12063e2-9afc-472a-b0d0-eb4eee4d9fb8
# ╟─44bdb36d-cf1e-4a5a-b97c-5b5d9f45b54a
# ╠═80a5d0d1-6522-4464-9eb4-bfd46b2124a1
# ╠═2b59335e-14d5-4fee-acc7-af1e5e458c09
# ╟─e1599782-86c1-443e-8f4f-59e47fdd5f86
# ╟─6e9b9800-3ea8-432c-905f-de2049a6021a
# ╠═60d1816d-5dd3-4166-bb11-8cd2307bc9ee
# ╟─de79e133-798a-490b-a4c9-9f38d5f31a98
# ╟─4d57266e-0800-4f95-ac49-3a2ba791bcb4
# ╟─5d80608d-74e3-4d0a-a65d-4231447f4ba3
# ╟─8aaf35b2-3c2c-45f9-afc0-2fe472f7cbbc
# ╠═02d8b06f-0add-4b9d-95da-25683e4ded82
# ╟─8d05a1b3-2501-47f2-abed-6362e1a6c946
# ╟─62535f41-dccc-46b9-9393-a833d6d1831b
# ╟─531136e8-2b27-4b76-aa4b-c84a9bcbe567
# ╟─27ce1f4d-4cb1-4fac-85b9-fdcb71cfb8f1
# ╠═c2ba23c6-d23b-4754-a5b8-371500420b43
# ╠═daccad47-5400-4a1b-b73d-f920e2bdc78a
# ╟─67d59347-2491-44f8-ad07-91f55e314d0e
# ╠═f00d2476-b96a-4595-a812-20cace428b75
# ╠═31b2fe2a-a778-4893-9188-c38d6e4b21e2
# ╠═9bf7d31b-7a4e-4ec9-99b9-d189c9b2297e
# ╠═1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
# ╠═c09828d7-82c0-40cf-ba0f-963ff1f7d8f7
# ╠═23c54f4b-6f03-4661-b9c2-5ace1028e577
# ╠═79f2fead-eff3-4a74-ac65-39dd5d51d361
# ╠═df5bec5f-fbb2-4f81-9398-c1b1ba317d1b
# ╠═2ef976be-ea40-446e-bbc0-3fb4e1c29dc7
# ╠═8b5e91c6-3355-4e23-abd5-3f67d4689c1c
# ╠═a1fc11c2-b0a9-422e-a2f0-cc28ff063c50
# ╠═03c6c2bf-364f-448e-969f-009fedaeb989
# ╠═195c6d1d-645e-4b7c-9d32-8f56359ebf99
# ╠═b425603d-e34d-4f95-a3d9-1bfce2982cfd
# ╠═05c64c67-e51a-49cc-8908-9d035fe18664
# ╠═ee682d4f-705d-43a8-b379-7b74af63792d
# ╠═9f49b73a-eea6-4533-971d-bafb322183e0
# ╟─c245c7c0-fbe0-4e6a-8e6e-91fc997bd9df
# ╠═a1d5177c-6540-475d-bae3-1cedfd692d7b
# ╠═f18be3eb-2007-46fe-9b9a-782796068f7f
# ╠═33ed15ce-e845-40d7-9423-4071f64f75d1
# ╠═bca24178-9793-4ab2-b7ab-861f9093e652
# ╠═5b954687-4cfc-4298-88f3-0b82e3877da8
# ╠═5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
# ╠═d62e9041-1b7a-4abe-8097-20d98fd3659a
# ╠═fc58665b-955b-4091-bd50-17ee0f8539ae
# ╠═ad8f309f-b3f1-4ed0-914f-675ffe23821f
# ╠═6f79058e-2243-4be1-aaf6-7e95d6f7c342
# ╠═05f14a3d-80a7-4ea6-8466-6f4d03fbe173
# ╠═ed8fe954-4a93-4e89-9667-ef73b8e3495e
# ╠═a35ca262-f5cc-404f-8c39-5d73e3477321
# ╟─218a9121-2633-493c-b126-5ddc5dcd8767
# ╟─00bf833f-22d5-47c3-ba59-6d7841bd93f0
# ╟─59cd2b6a-2f14-419f-be29-ef60b0f1c681
# ╠═7959b6c6-e501-4bd1-943d-fd8fd0efe4be
# ╠═902b2b49-5575-4b9f-aa7b-88583e213314
# ╠═7d2c5850-0ae9-44ea-ac31-130bc248e2ce
# ╠═b3a2f654-9e0e-4fcf-bf35-924e097427fe
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
