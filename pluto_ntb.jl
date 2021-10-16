### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
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
input_filename = "ED-00001-00000002.toc"

# ╔═╡ 76c03bc8-72b9-4fae-9310-3eb61d593896
@time parties, candidates, election = parse_data2(input_filename)

# ╔═╡ 228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
model_config = Spearman_model_config(
	log_name = "spearman", 
	weight_func = dist -> (1/2)^dist, 
	m = 30, 
	openmindedness_distr = Distributions.Normal(0.5, 0.1),
	stubbornness_distr = Distributions.Normal(0.5, 0.1)
)

# ╔═╡ 4a2b607d-947d-47e9-b73f-93eab1fb07a5
model = Spearman_model(election, length(candidates), model_config)

# ╔═╡ 4079d722-1201-4b10-a2a2-9aa420089d67
model.social_network

# ╔═╡ 26be9903-64f3-487f-af3f-dd1fc26c3665
Base.summarysize(model)

# ╔═╡ b5d93a4e-539f-499d-b0e3-7be9053a1572
exp_config = Exp_config(
    sample_size = 3000,
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
)

# ╔═╡ 90b8efaf-ce7e-4878-bf55-93f5e2a1b802
OpinionDiffusion.Plots.plotly()

# ╔═╡ 0a01a7ff-b15e-4039-bd16-c053b4b33f8e
experiment = Experiment(model, candidates, exp_config)

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diffusion_config = Diffusion_config(
        diffusions = 10,
        checkpoint = 100,
        voter_diff_config = Voter_diff_config(
            evolve_vertices = 1000,
			attract_proba = 0.4,
			change_rate = 0.5,
            method = "averageAll"
        ),
        edge_diff_config = Edge_diff_config(
            evolve_edges = 5000,
            dist_metric = Distances.Cityblock(),
            edge_diff_func = dist -> (1/2)^(dist+6.28)
        )
    )

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
diffusion_metrics = run_experiment!(experiment, candidates, diffusion_config)

# ╔═╡ 7f138d72-419a-4642-b163-6ec58ce42d24
OpinionDiffusion.metrics_vis(diffusion_metrics, candidates, parties)

# ╔═╡ 86659fc0-af7e-4498-8388-3e79349e9eb4
@bind step Slider(1 : length(diffusion_metrics.degree_distributions), show_value=true)

# ╔═╡ 43976886-9b44-4152-bb43-88e24f6c98f9

OpinionDiffusion.Plots.plot(

OpinionDiffusion.draw_voter_vis(
	diffusion_metrics.projections[step], diffusion_metrics.clusters[step], 				experiment.voter_visualization_config),
	
OpinionDiffusion.draw_degree_distr(diffusion_metrics.degree_distributions[step]),

OpinionDiffusion.draw_edge_distances(diffusion_metrics.edge_distances[step]), size = (1800,1920))

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
# ╠═4a2b607d-947d-47e9-b73f-93eab1fb07a5
# ╠═4079d722-1201-4b10-a2a2-9aa420089d67
# ╠═26be9903-64f3-487f-af3f-dd1fc26c3665
# ╠═b5d93a4e-539f-499d-b0e3-7be9053a1572
# ╠═90b8efaf-ce7e-4878-bf55-93f5e2a1b802
# ╠═0a01a7ff-b15e-4039-bd16-c053b4b33f8e
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╠═7f138d72-419a-4642-b163-6ec58ce42d24
# ╟─86659fc0-af7e-4498-8388-3e79349e9eb4
# ╠═43976886-9b44-4152-bb43-88e24f6c98f9
