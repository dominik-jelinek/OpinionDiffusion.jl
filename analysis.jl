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

# ╔═╡ 6ff48f11-656b-4084-acc7-fc7c4dba7d7c
import Distributions, Distances, Graphs, Plots, Random

# ╔═╡ 3d2caf1f-c781-40be-944f-f85b3f56a42a
Plots.plotly()
#Plots.gr()

# ╔═╡ e12063e2-9afc-472a-b0d0-eb4eee4d9fb8
md"## Configure visualizations"

# ╔═╡ c2ba23c6-d23b-4754-a5b8-371500420b43
model_log = load_log(logger.exp_dir, 0)

# ╔═╡ f00d2476-b96a-4595-a812-20cace428b75
begin
	sample_size = min(512, length(model.voters))
	sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)
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
else# dim_reduction_method == "MDS"
	dim_reduction_config = MDS_dim_reduction_config(out_dim)
end

# ╔═╡ de79e133-798a-490b-a4c9-9f38d5f31a98
md"""Clustering method: $(@bind clustering_method Select(["Kmeans", "GM", "Party", "DBSCAN"], default="Party"))"""

# ╔═╡ 60d1816d-5dd3-4166-bb11-8cd2307bc9ee
if clustering_method == "Party"
	clustering_config = Party_clustering_config(candidates)
elseif clustering_method == "Kmeans"
	clustering_config = Kmeans_clustering_config(length(candidates))
elseif clustering_method == "GM"
	clustering_config = GM_clustering_config(length(candidates))
else# clustering_method == "DBSCAN"
	clustering_config = DBSCAN_clustering_config(0.1, 3)
end

# ╔═╡ 1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
labels, clusters = clustering(sampled_voters, clustering_config)

# ╔═╡ 5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
title = dim_reduction_method * "_" * clustering_method * "_" * string(length(sampled_voters))

# ╔═╡ d3801426-fe49-4c46-81b5-0e55a673bc27
log_idxs = sort([parse(Int64, split(splitext(file)[1], "_")[end]) for file in readdir(logger.exp_dir) if split(file, "_")[1] == "model"])

# ╔═╡ ad8f309f-b3f1-4ed0-914f-675ffe23821f
visualizations = OpinionDiffusion.gather_vis2(logger.exp_dir, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ 7959b6c6-e501-4bd1-943d-fd8fd0efe4be
@bind clk Clock()

# ╔═╡ 902b2b49-5575-4b9f-aa7b-88583e213314
t = clk % length(visualizations) + 1

# ╔═╡ 7d2c5850-0ae9-44ea-ac31-130bc248e2ce
visualizations[t]

# ╔═╡ 50dd19a7-2aa7-4fe3-a350-315bc3f0dd62
visualizations["distances"][t + 1]

# ╔═╡ 4992d8eb-2c10-46ee-8bec-9002b35fc5fd
visualizations["voters"][t + 1]

# ╔═╡ 6bd911e6-1135-4562-b4be-284fc8d6886e
draw_degree_distr(Graphs.degree_histogram(model_log.social_network))

# ╔═╡ 86b6af14-ba5b-4710-8348-45965c616ce6
OpinionDiffusion.model_vis2(model, sampled_voter_ids, dim_reduction_config, clustering_config)

# ╔═╡ eacbf9e0-8191-46fc-ab5f-186a244a7aae
md"### Node visualization"

# ╔═╡ 846390ef-0817-48c6-889a-18fedf2a8df3
node_id = 42

# ╔═╡ 867261f8-89b6-42cb-a616-a4f3c553d522
depth = 1

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

# ╔═╡ Cell order:
# ╠═b6b67170-8dcf-11ed-22a8-c935c756f9b0
# ╠═521fad35-0852-48a1-93b0-7b8794544706
# ╠═f1bbf513-b421-492d-80b9-fe6b9c8373c1
# ╠═68da078f-4b9a-4254-9c71-6a6c31761963
# ╠═8d2cc22a-72fa-4657-a0a2-6b1d586ec465
# ╠═6ff48f11-656b-4084-acc7-fc7c4dba7d7c
# ╠═3d2caf1f-c781-40be-944f-f85b3f56a42a
# ╟─e12063e2-9afc-472a-b0d0-eb4eee4d9fb8
# ╠═c2ba23c6-d23b-4754-a5b8-371500420b43
# ╠═f00d2476-b96a-4595-a812-20cace428b75
# ╠═31b2fe2a-a778-4893-9188-c38d6e4b21e2
# ╠═e1599782-86c1-443e-8f4f-59e47fdd5f86
# ╠═80a5d0d1-6522-4464-9eb4-bfd46b2124a1
# ╠═2b59335e-14d5-4fee-acc7-af1e5e458c09
# ╟─de79e133-798a-490b-a4c9-9f38d5f31a98
# ╠═60d1816d-5dd3-4166-bb11-8cd2307bc9ee
# ╠═1e6ea6b0-6fb6-4097-aa76-a29edafe5b2a
# ╠═5d368a4f-d7ff-4ee7-a7b0-5c14fbe0fb48
# ╠═d3801426-fe49-4c46-81b5-0e55a673bc27
# ╠═ad8f309f-b3f1-4ed0-914f-675ffe23821f
# ╠═7959b6c6-e501-4bd1-943d-fd8fd0efe4be
# ╠═902b2b49-5575-4b9f-aa7b-88583e213314
# ╠═7d2c5850-0ae9-44ea-ac31-130bc248e2ce
# ╠═50dd19a7-2aa7-4fe3-a350-315bc3f0dd62
# ╠═4992d8eb-2c10-46ee-8bec-9002b35fc5fd
# ╠═6bd911e6-1135-4562-b4be-284fc8d6886e
# ╠═86b6af14-ba5b-4710-8348-45965c616ce6
# ╟─eacbf9e0-8191-46fc-ab5f-186a244a7aae
# ╠═846390ef-0817-48c6-889a-18fedf2a8df3
# ╠═867261f8-89b6-42cb-a616-a4f3c553d522
# ╠═1d09cd30-6119-4718-a4cc-385f57862fcc
# ╠═c2fa22d9-51ff-4db1-bd88-32c1f494473f
# ╠═64945f43-c9e7-4209-ad8a-4c81f0ac9b5a
# ╠═fbc0593b-917a-44e2-8e1f-691839851e5c
# ╠═94c32e1c-8c12-4b49-95b6-ad7dfdf2c827
# ╠═af31098c-bba9-42f2-8a72-e03f51f26ff3
