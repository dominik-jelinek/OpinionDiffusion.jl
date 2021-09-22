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

# ╔═╡ fc5a5935-8f4e-47ad-8568-70cd61656e06
input_filename = "ED-00001-00000002.toc"

# ╔═╡ 76c03bc8-72b9-4fae-9310-3eb61d593896
@time parties, candidates, election = parse_data2(input_filename)

# ╔═╡ 228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
model_config = Dict(
    "weight_func" => Dict(
        "type" => "exp",
        "base" => 1/2
    ),
    "m" => 30
)

# ╔═╡ 4a2b607d-947d-47e9-b73f-93eab1fb07a5
model = Spearman_model(election, length(candidates), model_config)

# ╔═╡ 4079d722-1201-4b10-a2a2-9aa420089d67
model.social_network

# ╔═╡ 26be9903-64f3-487f-af3f-dd1fc26c3665
model.log_dir

# ╔═╡ b5d93a4e-539f-499d-b0e3-7be9053a1572
exp_config = Dict(
    "sample_size" => 3000,
    "voter_visualization_config" => Dict(
        "used" => true,
        "reduce_dim_config" => Dict(
            "used" => true,
            "method" => "PCA",
            "PCA" => Dict(
                "out_dim" => 2
            ),
            "tsne" => Dict(
                "out_dim" => 2,
                "reduce_dims" => 0,
                "max_iter" => 3000,
                "perplexity" => 100.0
            )
        ),
        "clustering_config" => Dict(
            "used" => true,
            "method" => "Party",
            "K-means" => Dict(
                "cluster_count" => 8
            ),
            "GM" => Dict(
                "cluster_count" => 8
            )
        )
    )
)

# ╔═╡ 90b8efaf-ce7e-4878-bf55-93f5e2a1b802
begin
	OpinionDiffusion.gr()
	backend = OpinionDiffusion.Plots.GRBackend
end

# ╔═╡ bf41eeb9-121d-44d0-a95d-07ac0aaf5671


# ╔═╡ 0a01a7ff-b15e-4039-bd16-c053b4b33f8e
experiment = Experiment(model, candidates, parties, backend, exp_config)

# ╔═╡ f43b3b4c-9075-414b-9694-83e7c841605f
diffusion_config = Dict(
        "diffusions" => 100,
        "checkpoint" => 100,
        "voter_diff_config" => Dict(
            "evolve_vertices" => 10000,
			"attract_proba" => 0.8,
			"change_rate" => 0.5,
            "method" => "averageAll"
        ),
        "edge_diff_config" => Dict(
            "evolve_edges" => 10000,
            "dist_metric" => "L1",
            "edge_diff_func" => Dict(
                "type" => "exp",
        		"base" => 1/2,
        		"offset" => -6.28
            )
        )
    )

# ╔═╡ f6b4ba47-f9d2-42f0-9c86-e9810be7b810
diffusion_metrics, visualizations = run_experiment!(experiment, candidates, parties, diffusion_config)

# ╔═╡ 7f138d72-419a-4642-b163-6ec58ce42d24
visualize_metrics(diffusion_metrics, candidates, parties, experiment.exp_dir)

# ╔═╡ 86659fc0-af7e-4498-8388-3e79349e9eb4
@bind step Slider(1 : length(visualizations.degree_distributions), show_value=true)

# ╔═╡ 43976886-9b44-4152-bb43-88e24f6c98f9

OpinionDiffusion.plot(experiment.visualizations.voter_visualizations[step], experiment.visualizations.degree_distributions[step], layout = (2, 1), size = (980,1200))

# ╔═╡ Cell order:
# ╠═86c32615-cdba-41aa-bfca-e5b90563f7f7
# ╠═481f2e27-0f88-482a-846a-6a31bf38f3ba
# ╠═9284976e-d474-11eb-2b94-dbe906a08bd7
# ╠═09ed6ee3-c46e-4223-9757-bb01d58f13f4
# ╠═f70ddba7-3f23-4f60-9a70-fb4c7d5ff791
# ╠═fc5a5935-8f4e-47ad-8568-70cd61656e06
# ╠═76c03bc8-72b9-4fae-9310-3eb61d593896
# ╠═228f2e5e-cf91-4c00-9c92-6ebbcdc4c69a
# ╠═4a2b607d-947d-47e9-b73f-93eab1fb07a5
# ╠═4079d722-1201-4b10-a2a2-9aa420089d67
# ╠═26be9903-64f3-487f-af3f-dd1fc26c3665
# ╠═b5d93a4e-539f-499d-b0e3-7be9053a1572
# ╠═90b8efaf-ce7e-4878-bf55-93f5e2a1b802
# ╠═bf41eeb9-121d-44d0-a95d-07ac0aaf5671
# ╠═0a01a7ff-b15e-4039-bd16-c053b4b33f8e
# ╠═f43b3b4c-9075-414b-9694-83e7c841605f
# ╠═f6b4ba47-f9d2-42f0-9c86-e9810be7b810
# ╟─7f138d72-419a-4642-b163-6ec58ce42d24
# ╟─86659fc0-af7e-4498-8388-3e79349e9eb4
# ╟─43976886-9b44-4152-bb43-88e24f6c98f9
