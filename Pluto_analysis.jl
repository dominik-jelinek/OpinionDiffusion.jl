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

# ╔═╡ 249c2c6d-22c7-4951-b067-ba1bf27573c3
using Statistics

# ╔═╡ 6076b5d6-9fc0-47ba-8d90-059f80d941c9
using GraphMakie, CairoMakie

# ╔═╡ 6ff48f11-656b-4084-acc7-fc7c4dba7d7c
import Distributions, Distances, Graphs, Plots, Random

# ╔═╡ 3d2caf1f-c781-40be-944f-f85b3f56a42a
Plots.plotly()
#Plots.gr()

# ╔═╡ 2dad0bf5-6d17-4bf5-92ca-c77421d0dba3
TableOfContents(depth=4)

# ╔═╡ 5d80608d-74e3-4d0a-a65d-4231447f4ba3
md"### Loading data"

# ╔═╡ 8aaf35b2-3c2c-45f9-afc0-2fe472f7cbbc
log_dir_path = "./logs"

# ╔═╡ 02d8b06f-0add-4b9d-95da-25683e4ded82
md"""Select experiment: $(@bind experiment_dir Select([filename for filename in readdir(log_dir_path) if isdir(log_dir_path * "/" * filename)]))"""

# ╔═╡ 8d05a1b3-2501-47f2-abed-6362e1a6c946
exp_dir_path = log_dir_path * "/" * experiment_dir

# ╔═╡ e50263d1-1800-4065-bc47-e2559949b5c7
init_model = load_model(exp_dir_path, 0)

# ╔═╡ 02a148bb-38f7-4fc4-9ae0-b7ef2e3b9b15
experiment_config = load_config(exp_dir_path)

# ╔═╡ 5a82bfad-6d86-469b-8229-ffb7872e63db
candidates = get_candidates(init_model)

# ╔═╡ 67d59347-2491-44f8-ad07-91f55e314d0e
md"### Sampling"

# ╔═╡ f00d2476-b96a-4595-a812-20cace428b75
begin
	sample_size = length(init_model.voters)#min(1024, length(init_model.voters))
	sampled_voter_ids = sort(OpinionDiffusion.StatsBase.sample(1:length(init_model.voters), sample_size, replace=false))
end

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
        											reduce_dims = 5,
        											max_iter = 2000,
        											perplexity = 50.0)
else dim_reduction_method == "MDS"
	dim_reduction_config = MDS_dim_reduction_config(out_dim)
end

# ╔═╡ bfb16d83-d0ca-4201-9c47-3584d23532f5
init_projections = reduce_dims(get_voters(init_model), dim_reduction_config)

# ╔═╡ 6e9b9800-3ea8-432c-905f-de2049a6021a
md"### Clustering"

# ╔═╡ de79e133-798a-490b-a4c9-9f38d5f31a98
md"""Clustering method: $(@bind clustering_method Select(["Kmeans", "GM", "Party", "DBSCAN", "Density"], default="Party"))"""

# ╔═╡ 60d1816d-5dd3-4166-bb11-8cd2307bc9ee
if clustering_method == "Party"
	clustering_config = Party_clustering_config(candidates)
elseif clustering_method == "Kmeans"
	clustering_config = Kmeans_clustering_config(length(candidates))
elseif clustering_method == "GM"
	clustering_config = GM_clustering_config(length(candidates))
elseif clustering_method == "DBSCAN"
	clustering_config = DBSCAN_clustering_config(0.02, 10)
else
	clustering_config = Density_clustering_config(3)
end

# ╔═╡ 5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
title = dim_reduction_method * "_" * clustering_method * "_" * string(sample_size)

# ╔═╡ 00bf833f-22d5-47c3-ba59-6d7841bd93f0
md"## Evolving visualizations"

# ╔═╡ c2bf6676-5922-4e8c-ba3e-2331cdb07725
md"""
---
**Execution barrier** $(@bind gather_vis_eb CheckBox())

---
"""

# ╔═╡ ad8f309f-b3f1-4ed0-914f-675ffe23821f
if gather_vis_eb
	visualizations = gather_vis(exp_dir_path, sampled_voter_ids, dim_reduction_config, clustering_config)	
end

# ╔═╡ 7959b6c6-e501-4bd1-943d-fd8fd0efe4be
@bind clk Clock()

# ╔═╡ 6831b137-faec-42fc-aea7-fbc70b663dbc
t = clk % length(visualizations) + 1

# ╔═╡ 52f8f2b0-4a21-4a05-8f25-62175ba6de07
visualizations[t][1]

# ╔═╡ fc6982c4-c79d-4d1d-ad1a-3888943629bc
#visualizations[t][2]

# ╔═╡ 7f136b0d-2d39-48a4-89e6-96fc7b574bc1
visualizations[t][3]

# ╔═╡ 9e2c288a-47bb-4b43-9668-fdd5ac22f43e
#visualizations[t][4]

# ╔═╡ 7d2c5850-0ae9-44ea-ac31-130bc248e2ce
#visualizations[t][5]

# ╔═╡ 4d57266e-0800-4f95-ac49-3a2ba791bcb4
md"## Specific Diffusion Step"

# ╔═╡ a0449758-593e-403b-a3ab-352a2bc3717b
md"""Select diffusion step: $(@bind diff_step PlutoUI.Slider(sort([parse(Int64, split(filename, "_")[2][1:end-5]) for filename in readdir(exp_dir_path) if split(filename, "_")[1] == "model"])))"""

# ╔═╡ c3ba833f-9eee-46c6-b7ae-ea8cb831b6dc
diff_step

# ╔═╡ c2ba23c6-d23b-4754-a5b8-371500420b43
model_log = load_model(exp_dir_path, diff_step)

# ╔═╡ 41deda04-850e-4463-9161-f768e4631c90
summary = get_election_summary(get_votes(get_voters(model_log)), drop_last=true)

# ╔═╡ 8d407120-caae-4991-a250-ac84cc9148c4
draw_election_summary(summary)

# ╔═╡ f35712d1-3c91-4202-8e90-55b1922dbd8e
vis, _ = timestamp_vis(model_log, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ 230b0583-95ba-4dea-8593-81473a936f9e
vis[1]

# ╔═╡ e332e426-eb63-4a69-a330-7b0127ad1a2c
vis[2]

# ╔═╡ 6b658e1e-d4c4-490a-83d1-40a0070db598
vis[3]

# ╔═╡ 996e9336-8175-4ddb-aa37-9ef7626a4f74
vis[4]

# ╔═╡ b37105e0-776c-44d2-936d-5587b8d9859d
vis[5]

# ╔═╡ 581f20a7-6393-41ea-8fe7-510b6c8a1e89
begin
	voters = get_voters(model_log)
	social_network = get_social_network(model_log)
end

# ╔═╡ 31b2fe2a-a778-4893-9188-c38d6e4b21e2
sampled_voters = voters[sampled_voter_ids]

# ╔═╡ 9bf7d31b-7a4e-4ec9-99b9-d189c9b2297e
begin
	projections = reduce_dims(sampled_voters, dim_reduction_config)
	unify_projections!(init_projections, projections)
	projections
end

# ╔═╡ 1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
labels, clusters = clustering(sampled_voters, clustering_config, projections)

# ╔═╡ 23c54f4b-6f03-4661-b9c2-5ace1028e577
dists = get_distance(sampled_voters)

# ╔═╡ df5bec5f-fbb2-4f81-9398-c1b1ba317d1b
for i in eachindex(clusters)
	println(clusters[i][1], ": ", mean(dists[1, collect(clusters[i][2])]))
end

# ╔═╡ a1fc11c2-b0a9-422e-a2f0-cc28ff063c50
sort(unique(labels))

# ╔═╡ 2ef976be-ea40-446e-bbc0-3fb4e1c29dc7
function remove_holes(labels::Vector{Int64})
    unique_labels = sort(unique(labels)) # get the unique labels
	mapping = Dict(value => i for (i, value) in enumerate(unique_labels))
	return mapping
end

# ╔═╡ 03c6c2bf-364f-448e-969f-009fedaeb989
mapping = remove_holes(labels)

# ╔═╡ 195c6d1d-645e-4b7c-9d32-8f56359ebf99
new_labels = [mapping[label] for label in labels]

# ╔═╡ b425603d-e34d-4f95-a3d9-1bfce2982cfd
sils = OpinionDiffusion.Clustering.silhouettes(new_labels, dists)

# ╔═╡ 05c64c67-e51a-49cc-8908-9d035fe18664
median(sils)

# ╔═╡ c245c7c0-fbe0-4e6a-8e6e-91fc997bd9df
 #md"Diff: $(@bind diff PlutoUI.Slider(1:100))"

# ╔═╡ 218a9121-2633-493c-b126-5ddc5dcd8767
md"### Compounded metrics"

# ╔═╡ 59cd2b6a-2f14-419f-be29-ef60b0f1c681
md"## Specific Community"

# ╔═╡ e02a5e33-75a6-4bb3-9a45-8ac0dedb3f5c
@bind cluster_id PlutoUI.Slider(1:length(clusters))

# ╔═╡ 5d0a8c5f-fc2a-48c1-92d6-7e0aaf52ad9e
cluster_id

# ╔═╡ 8d700326-62c9-4775-bb86-9a1393b9c87c
[(cl[1], reverse(sortperm(borda_voting(get_votes(get_voters(model_log)[collect(cl[2])]), length(get_candidates(model_log)), true)))) for cl in clusters]

# ╔═╡ de27090f-d315-479a-ba17-9d4bc67342ca
[length(cl[2]) for cl in clusters]

# ╔═╡ 059b1554-b27f-4c59-a99f-2ad6b93b9366
cluster = clusters[cluster_id]

# ╔═╡ eacbf9e0-8191-46fc-ab5f-186a244a7aae
md"## Specific Node"

# ╔═╡ 846390ef-0817-48c6-889a-18fedf2a8df3
@bind node_id PlutoUI.Slider(1:length(voters))

# ╔═╡ 4c07d708-4da2-449c-b3e7-115e35fb6d58
node_id

# ╔═╡ 867261f8-89b6-42cb-a616-a4f3c553d522
depth = 1

# ╔═╡ a737af01-3210-4340-b7c9-99a9060b295b
md"## Helper functions"

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

    return Graphs.induced_subgraph(social_network, sort(collect(ego_nodes)))
end

# ╔═╡ 1d09cd30-6119-4718-a4cc-385f57862fcc
ego_subgraph, ego_ids = ego(get_social_network(model_log), node_id, depth)

# ╔═╡ 64945f43-c9e7-4209-ad8a-4c81f0ac9b5a
ego_project = projections[:, ego_ids]

# ╔═╡ fbc0593b-917a-44e2-8e1f-691839851e5c
x, y = ego_project[1, :], ego_project[2, :]

# ╔═╡ 0c59f190-d1b3-4bf9-8577-7fad3e53143a
begin
	f = Figure()
	colors = OpinionDiffusion.Colors.distinguishable_colors(maximum([cluster[1] for cluster in clusters]))
	node_colors = colors[labels[ego_ids]]
	nodesizes=[Graphs.degree(get_social_network(model_log), node) for node in ego_ids]
	nodelabels = [string(id) for id in ego_ids]
	graphplot(f[1,1], ego_subgraph, layout=ego_subgraph -> Point.(zip(x, y)), node_color=node_colors, nlabels=nodelabels)
	f
end

# ╔═╡ fc865475-463f-40b1-9481-35123adcbffa
[sortperm(borda_voting(get_votes(sampled_voters[collect(cluster)]), length(candidates), true), rev=true) for cluster in clusters if length(cluster) != 0]

# ╔═╡ 2750e431-7c7f-4323-b74b-399ab7346603
countss = [get_counts(get_votes(sampled_voters[collect(cluster)]), length(candidates)) for cluster in clusters if length(cluster) != 0]

# ╔═╡ fc58665b-955b-4091-bd50-17ee0f8539ae
plots = Plots.plot([Plots.heatmap(count, yticks=1:length(src_candidates), xticks=1:length(src_candidates), ylabel="Candidate", xlabel="Position") for count in countss]..., layout = (length(candidates), 1), size = (669,900))

# ╔═╡ 816b8848-6198-498f-bacf-790693babfbc


# ╔═╡ Cell order:
# ╠═b6b67170-8dcf-11ed-22a8-c935c756f9b0
# ╠═521fad35-0852-48a1-93b0-7b8794544706
# ╠═f1bbf513-b421-492d-80b9-fe6b9c8373c1
# ╠═68da078f-4b9a-4254-9c71-6a6c31761963
# ╠═8d2cc22a-72fa-4657-a0a2-6b1d586ec465
# ╠═249c2c6d-22c7-4951-b067-ba1bf27573c3
# ╠═6ff48f11-656b-4084-acc7-fc7c4dba7d7c
# ╠═6076b5d6-9fc0-47ba-8d90-059f80d941c9
# ╠═3d2caf1f-c781-40be-944f-f85b3f56a42a
# ╟─2dad0bf5-6d17-4bf5-92ca-c77421d0dba3
# ╟─5d80608d-74e3-4d0a-a65d-4231447f4ba3
# ╟─8aaf35b2-3c2c-45f9-afc0-2fe472f7cbbc
# ╠═02d8b06f-0add-4b9d-95da-25683e4ded82
# ╟─8d05a1b3-2501-47f2-abed-6362e1a6c946
# ╠═e50263d1-1800-4065-bc47-e2559949b5c7
# ╠═02a148bb-38f7-4fc4-9ae0-b7ef2e3b9b15
# ╠═5a82bfad-6d86-469b-8229-ffb7872e63db
# ╟─67d59347-2491-44f8-ad07-91f55e314d0e
# ╠═f00d2476-b96a-4595-a812-20cace428b75
# ╠═bfb16d83-d0ca-4201-9c47-3584d23532f5
# ╟─e12063e2-9afc-472a-b0d0-eb4eee4d9fb8
# ╟─44bdb36d-cf1e-4a5a-b97c-5b5d9f45b54a
# ╠═80a5d0d1-6522-4464-9eb4-bfd46b2124a1
# ╠═2b59335e-14d5-4fee-acc7-af1e5e458c09
# ╟─e1599782-86c1-443e-8f4f-59e47fdd5f86
# ╟─6e9b9800-3ea8-432c-905f-de2049a6021a
# ╠═60d1816d-5dd3-4166-bb11-8cd2307bc9ee
# ╟─de79e133-798a-490b-a4c9-9f38d5f31a98
# ╠═5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
# ╟─00bf833f-22d5-47c3-ba59-6d7841bd93f0
# ╟─c2bf6676-5922-4e8c-ba3e-2331cdb07725
# ╠═ad8f309f-b3f1-4ed0-914f-675ffe23821f
# ╠═7959b6c6-e501-4bd1-943d-fd8fd0efe4be
# ╠═6831b137-faec-42fc-aea7-fbc70b663dbc
# ╠═52f8f2b0-4a21-4a05-8f25-62175ba6de07
# ╠═fc6982c4-c79d-4d1d-ad1a-3888943629bc
# ╠═7f136b0d-2d39-48a4-89e6-96fc7b574bc1
# ╠═9e2c288a-47bb-4b43-9668-fdd5ac22f43e
# ╠═7d2c5850-0ae9-44ea-ac31-130bc248e2ce
# ╟─4d57266e-0800-4f95-ac49-3a2ba791bcb4
# ╟─a0449758-593e-403b-a3ab-352a2bc3717b
# ╟─c3ba833f-9eee-46c6-b7ae-ea8cb831b6dc
# ╠═c2ba23c6-d23b-4754-a5b8-371500420b43
# ╠═41deda04-850e-4463-9161-f768e4631c90
# ╠═8d407120-caae-4991-a250-ac84cc9148c4
# ╠═f35712d1-3c91-4202-8e90-55b1922dbd8e
# ╠═230b0583-95ba-4dea-8593-81473a936f9e
# ╠═e332e426-eb63-4a69-a330-7b0127ad1a2c
# ╠═6b658e1e-d4c4-490a-83d1-40a0070db598
# ╠═996e9336-8175-4ddb-aa37-9ef7626a4f74
# ╠═b37105e0-776c-44d2-936d-5587b8d9859d
# ╟─581f20a7-6393-41ea-8fe7-510b6c8a1e89
# ╠═31b2fe2a-a778-4893-9188-c38d6e4b21e2
# ╟─9bf7d31b-7a4e-4ec9-99b9-d189c9b2297e
# ╠═1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
# ╟─23c54f4b-6f03-4661-b9c2-5ace1028e577
# ╠═df5bec5f-fbb2-4f81-9398-c1b1ba317d1b
# ╠═a1fc11c2-b0a9-422e-a2f0-cc28ff063c50
# ╠═2ef976be-ea40-446e-bbc0-3fb4e1c29dc7
# ╠═03c6c2bf-364f-448e-969f-009fedaeb989
# ╠═195c6d1d-645e-4b7c-9d32-8f56359ebf99
# ╠═b425603d-e34d-4f95-a3d9-1bfce2982cfd
# ╠═05c64c67-e51a-49cc-8908-9d035fe18664
# ╟─c245c7c0-fbe0-4e6a-8e6e-91fc997bd9df
# ╠═fc58665b-955b-4091-bd50-17ee0f8539ae
# ╟─218a9121-2633-493c-b126-5ddc5dcd8767
# ╟─59cd2b6a-2f14-419f-be29-ef60b0f1c681
# ╠═e02a5e33-75a6-4bb3-9a45-8ac0dedb3f5c
# ╠═5d0a8c5f-fc2a-48c1-92d6-7e0aaf52ad9e
# ╠═8d700326-62c9-4775-bb86-9a1393b9c87c
# ╠═de27090f-d315-479a-ba17-9d4bc67342ca
# ╠═059b1554-b27f-4c59-a99f-2ad6b93b9366
# ╟─eacbf9e0-8191-46fc-ab5f-186a244a7aae
# ╠═846390ef-0817-48c6-889a-18fedf2a8df3
# ╠═4c07d708-4da2-449c-b3e7-115e35fb6d58
# ╠═867261f8-89b6-42cb-a616-a4f3c553d522
# ╠═1d09cd30-6119-4718-a4cc-385f57862fcc
# ╠═64945f43-c9e7-4209-ad8a-4c81f0ac9b5a
# ╠═fbc0593b-917a-44e2-8e1f-691839851e5c
# ╠═0c59f190-d1b3-4bf9-8577-7fad3e53143a
# ╟─a737af01-3210-4340-b7c9-99a9060b295b
# ╠═af31098c-bba9-42f2-8a72-e03f51f26ff3
# ╠═fc865475-463f-40b1-9481-35123adcbffa
# ╠═2750e431-7c7f-4323-b74b-399ab7346603
# ╠═816b8848-6198-498f-bacf-790693babfbc
