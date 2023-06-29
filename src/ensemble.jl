@kwdef struct Ensemble_config
	input_filename::String
	selection_configs::Vector{Union{Selection_config, Nothing}}

	voter_configs::Vector{Abstract_voter_config}
	graph_configs::Vector{Abstract_graph_config}

	diffusion_init_configs::Union{Vector{Vector{Abstract_mutation_init_config}}, Nothing}
	diffusion_configs::Union{Vector{Diffusion_config}, Nothing}
end

function Base.length(ensemble_config::Ensemble_config)
	len = 1
	if length(ensemble_config.selection_configs) > 0
		len *= length(ensemble_config.selection_configs)
	end

	len *= length(ensemble_config.voter_init_configs)
	len *= length(ensemble_config.graph_init_configs)

	if length(ensemble_config.diff_init_configs) > 0
		len *= length(ensemble_config.diff_init_configs)
	end

	if length(ensemble_config.diff_configs) > 0
		len *= length(ensemble_config.diff_configs)
	end

	return len
end

function ensemble(init_election::Election, ensemble_config::Ensemble_config, get_metrics::Function)
	dataframes = Vector{DataFrame}()

	# election
	for (i, selection_config) in enumerate(ensemble_config.selection_configs)
		election = select(init_election, selection_config)

		prev_configs = Dict()
		prev_configs["selection_config"] = selection_config
		# model
		for (j, voter_init_config) in enumerate(ensemble_config.voter_configs)
			voter_init_config = resolve_dependencies(voter_init_config, prev_configs)
			prev_configs["voter_config"] = voter_init_config

			voters = init_voters(election.votes, voter_init_config)

			for (k, graph_init_config) in enumerate(ensemble_config.graph_configs)
				graph_init_config = resolve_dependencies(graph_init_config, prev_configs)
				prev_configs["graph_config"] = graph_init_config

				social_network = init_graph(voters, graph_init_config)
				model = General_model(voters, social_network, election.party_names, election.candidates)

				accumulator = Accumulator(get_metrics)
				add_metrics!(accumulator, model)

				# no diffusion
				if ensemble_config.diffusion_configs === nothing
					df = hcat(DataFrame(prev_configs), accumulated_metrics(accumulator))
					df.diffusion_step = [0]
					push!(dataframes, df)
					continue
				end

				# diffusion
				for (l, diffusion_init_config) in enumerate(ensemble_config.diffusion_init_configs)
					for n in eachindex(diffusion_init_config)
						diffusion_init_config[n] = resolve_dependencies(diffusion_init_config[n], prev_configs)
					end
					prev_configs["diffusion_init_config"] = diffusion_init_config

					model_init = deepcopy(model)
					init_diffusion!(model_init, diffusion_init_config)

					for (m, diffusion_config) in enumerate(ensemble_config.diffusion_configs)
						for n in eachindex(diffusion_config.mutation_configs)
							diffusion_config.mutation_configs[n] = resolve_dependencies(diffusion_config.mutation_configs[n], prev_configs)
						end
						prev_configs["diffusion_config"] = diffusion_config

						accumulator_diffusion = deepcopy(accumulator)
						model_diffusion = deepcopy(model_init)
						run!(model_diffusion, diffusion_config; accumulator=accumulator_diffusion)

						expanded_configs = Dict(key => fill(value, diffusion_config.diffusion_steps + 1) for (key, value) in prev_configs)

						df = hcat(DataFrame(expanded_configs), accumulated_metrics(accumulator_diffusion))
						df.diffusion_step = collect(0:diffusion_config.diffusion_steps)

						push!(dataframes, df)

						delete!(prev_configs, "diffusion_config")
					end

					delete!(prev_configs, "diffusion_init_config")
				end

				delete!(prev_configs, "graph_config")
			end

			delete!(prev_configs, "voter_config")
		end
	end

	return vcat(dataframes...)
end

@kwdef struct Experiment_config
	input_filename::String
	selection_config::Union{Selection_config, Nothing}

	voter_config::Abstract_voter_config
	graph_config::Abstract_graph_config

	diffusion_init_config::Union{Vector{Abstract_mutation_init_config}, Nothing}
	diffusion_config::Union{Diffusion_config, Nothing}
end

function resolve_dependencies(config::Abstract_config, prev_configs)
	return config
end