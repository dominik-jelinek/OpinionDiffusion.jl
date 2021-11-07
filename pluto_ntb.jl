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
using Revise

# ╔═╡ 09ed6ee3-c46e-4223-9757-bb01d58f13f4
using PlutoUI

# ╔═╡ f70ddba7-3f23-4f60-9a70-fb4c7d5ff791
using OpinionDiffusion

# ╔═╡ 75588cfd-52c9-4406-975c-c03158db6e78
import Distributions, Distances

# ╔═╡ fc5a5935-8f4e-47ad-8568-70cd61656e06
input_filename = "madeUp"#"ED-00001-00000002.toc"

# ╔═╡ 76c03bc8-72b9-4fae-9310-3eb61d593896
@time parties, candidates, election = parse_data2(input_filename)

# ╔═╡ 228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
model_config = General_model_config( 
	m = 2,
	voter_config = Spearman_voter_config(
		weight_func = dist -> (1/2)^dist, 
		openmindedness_distr = Distributions.Normal(0.5, 0.1),
		stubbornness_distr = Distributions.Normal(0.5, 0.1)
	)
)

# ╔═╡ 985131a7-7c11-4f9d-ae00-ef031002592d
model_dir = "logs/model_2021-10-18_21-46-20"

# ╔═╡ d8b613c5-7276-466d-a54a-7670f0921b35
exp_dir = model_dir * "/" * "experiment_2021-10-18_22-06-26"

# ╔═╡ 571e7a33-20b7-4432-b553-c07b9081c68d
idx = -1

# ╔═╡ 1937642e-7f63-4ffa-b01f-22208a716dac
md"""
Barrier $(@bind cb_barrier CheckBox())
"""

# ╔═╡ 72450aaf-c6c4-458e-9555-39c31345116b
md"""
Model source $(@bind model_source Select(["restart_model" => "Restart", "load_model" => "Load", "new_model" => "New (takes time)"]))
"""

# ╔═╡ 4a2b607d-947d-47e9-b73f-93eab1fb07a5
if cb_barrier 
	if model_source == "new_model"
		model = General_model(election, length(candidates), model_config)
		model_states = [model]
		logger = Logger(model)
	elseif model_source == "load_model"
		model = load_log(exp_dir, idx)
		model_states = load_logs(exp_dir, 0, idx)
		logger = Logger(model, model_dir, exp_dir, idx)
	else #restart
		model = load_log(model_dir)
		model_states = [model]
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
        diffusions = 10,
        checkpoint = 1,
        voter_diff_config = Voter_diff_config(
            evolve_vertices = 1000,
			attract_proba = 0.8,
			change_rate = 0.5,
            method = "averageAll"
        ),
        edge_diff_config = Edge_diff_config(
            evolve_edges = 5000,
            dist_metric = Distances.Cityblock(),
            edge_diff_func = dist -> (1/2)^(dist+6.28)
        )
    )

# ╔═╡ 20819900-1129-4ff1-b97e-d079ffce8ab8
md"""
Run experiment $(@bind cb CheckBox())
"""

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
if cb
	models = run!(model, diffusion_config, logger)
end

# ╔═╡ ff873978-d93f-4ba2-aadd-6cfd3b136e3d
OpinionDiffusion.Plots.plotly()

# ╔═╡ a85f9e15-453a-4387-a214-194dc019ef0f


# ╔═╡ 688e9359-623d-467e-843d-033498e84c29
function update_metrics!(model, diffusion_metrics, candidates)
    dict = degree_histogram(model.social_network)
    keyss = collect(keys(dict))
    push!(diffusion_metrics.min_degrees, minimum(keyss))
    push!(diffusion_metrics.avg_degrees, LightGraphs.ne(model.social_network) * 2 / LightGraphs.nv(model.social_network))
    push!(diffusion_metrics.max_degrees, maximum(keyss))
    
    votes = get_votes(model.voters)
    can_count = length(candidates)
    push!(diffusion_metrics.plurality_votings, plurality_voting(votes, can_count, true))
    push!(diffusion_metrics.borda_votings, borda_voting(votes, can_count, true))
    push!(diffusion_metrics.copeland_votings, copeland_voting(votes, can_count))
end

# ╔═╡ c39a365b-926a-4759-b1a6-7ff85ae032a0
for model in models
	update_metrics!(model)
end

# ╔═╡ 97ea949c-d291-43cb-8f71-1fba22560e1e
voter_vis_config = Voter_vis_config(
        used = true,
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
        ),
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
)

# ╔═╡ 228f2d0b-b964-4133-b0bc-fee7d9fe298f
sample_size = 5

# ╔═╡ ea2d9183-07d1-454b-8e0a-1715528dc13e
sampled_voter_ids = OpinionDiffusion.StatsBase.sample(1:length(model.voters), sample_size, replace=false)

# ╔═╡ b8be568b-8cc1-47a1-83b8-a5b423a7b08d
projections, labels, clusters = get_voter_vis(model.voters, sampled_voter_ids, candidates, voter_visualization_config)

# ╔═╡ 4eb56ee4-04e9-40cd-90fc-1495215f2928
OpinionDiffusion.draw_voter_vis(projections, clusters, voter_visualization_config)

# ╔═╡ 83b3e762-0aeb-4037-a723-d66f20540c2c
Base.summarysize(models[10])

# ╔═╡ d4896d75-6c3e-4a7d-ac83-7d6c255da941
sum(OpinionDiffusion.get_opinion(modelss[1].voters[29988]))

# ╔═╡ 7f138d72-419a-4642-b163-6ec58ce42d24
OpinionDiffusion.metrics_vis(diffusion_metrics, candidates, parties)

# ╔═╡ 86659fc0-af7e-4498-8388-3e79349e9eb4
@bind step Slider(1 : length(diffusion_metrics.degree_distributions), show_value=true)

# ╔═╡ 43976886-9b44-4152-bb43-88e24f6c98f9

OpinionDiffusion.Plots.plot(

OpinionDiffusion.draw_voter_vis(
	diffusion_metrics.projections[step], diffusion_metrics.clusters[step], 				experiment.voter_visualization_config),
	
OpinionDiffusion.draw_degree_distr(LightGraphs.degree_histogram(model.social_network)),

OpinionDiffusion.draw_edge_distances(get_edge_distances(model.social_network, model.voters)), size = (1800,1920))

# ╔═╡ 6e8ec73b-59b7-4bf5-8c32-0c8b6911fef9
function Spearman_metrics(model, can_count)
    dict = LightGraphs.degree_histogram(model.social_network)
    keyss = collect(keys(dict))
    votes = get_votes(model.voters)

    return Spearman_metrics([minimum(keyss)], 
                            [LightGraphs.ne(model.social_network) * 2 / LightGraphs.LightGraphs.nv(model.social_network)], 
                            [maximum(keyss)], 
                            [plurality_voting(votes, can_count, true)], 
                            [borda_voting(votes, can_count, true)], 
                            [copeland_voting(votes, can_count)]
                            )
end

# ╔═╡ 93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
struct Spearman_metrics
    #min_distance::Vector{Int64}
    #avg_distance::Vector{Float64}
    #max_distance::Vector{Int64}
    
    #graph metrics
    min_degrees::Vector{Int}
    avg_degrees::Vector{Float64}
    max_degrees::Vector{Int}
 
    #election results
    plurality_votings::Vector{Vector{Float64}}
    borda_votings::Vector{Vector{Float64}}
    copeland_votings::Vector{Vector{Float64}}
    #STV
end

# ╔═╡ Cell order:
# ╠═86c32615-cdba-41aa-bfca-e5b90563f7f7
# ╠═481f2e27-0f88-482a-846a-6a31bf38f3ba
# ╠═9284976e-d474-11eb-2b94-dbe906a08bd7
# ╠═09ed6ee3-c46e-4223-9757-bb01d58f13f4
# ╠═f70ddba7-3f23-4f60-9a70-fb4c7d5ff791
# ╠═75588cfd-52c9-4406-975c-c03158db6e78
# ╠═fc5a5935-8f4e-47ad-8568-70cd61656e06
# ╠═76c03bc8-72b9-4fae-9310-3eb61d593896
# ╠═228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
# ╠═776f99ea-6e44-4eb6-b2dd-3552bbb39954
# ╠═985131a7-7c11-4f9d-ae00-ef031002592d
# ╠═d8b613c5-7276-466d-a54a-7670f0921b35
# ╠═571e7a33-20b7-4432-b553-c07b9081c68d
# ╟─1937642e-7f63-4ffa-b01f-22208a716dac
# ╟─72450aaf-c6c4-458e-9555-39c31345116b
# ╠═4a2b607d-947d-47e9-b73f-93eab1fb07a5
# ╠═4079d722-1201-4b10-a2a2-9aa420089d67
# ╠═26be9903-64f3-487f-af3f-dd1fc26c3665
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╟─20819900-1129-4ff1-b97e-d079ffce8ab8
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═ff873978-d93f-4ba2-aadd-6cfd3b136e3d
# ╠═c39a365b-926a-4759-b1a6-7ff85ae032a0
# ╠═93aa822a-1be4-45c0-a6a0-65a3d8f08bbf
# ╠═6e8ec73b-59b7-4bf5-8c32-0c8b6911fef9
# ╠═a85f9e15-453a-4387-a214-194dc019ef0f
# ╠═688e9359-623d-467e-843d-033498e84c29
# ╠═97ea949c-d291-43cb-8f71-1fba22560e1e
# ╠═228f2d0b-b964-4133-b0bc-fee7d9fe298f
# ╠═ea2d9183-07d1-454b-8e0a-1715528dc13e
# ╠═b8be568b-8cc1-47a1-83b8-a5b423a7b08d
# ╠═4eb56ee4-04e9-40cd-90fc-1495215f2928
# ╠═83b3e762-0aeb-4037-a723-d66f20540c2c
# ╠═d4896d75-6c3e-4a7d-ac83-7d6c255da941
# ╠═7f138d72-419a-4642-b163-6ec58ce42d24
# ╟─86659fc0-af7e-4498-8388-3e79349e9eb4
# ╠═43976886-9b44-4152-bb43-88e24f6c98f9
