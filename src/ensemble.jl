@kwdef struct Ensemble_config
	data_path::String
	remove_candidate_ids::Union{Vector{Int64}, Nothing} = nothing
	sampling_configs::Vector{Union{Sampling_config, Nothing}} = [nothing]

	voter_configs::Vector{Abstract_voter_config}
	graph_configs::Vector{Abstract_graph_config}

	diffusion_init_configs::Union{Vector{Vector{Abstract_mutation_init_config}}, Nothing} = nothing
	diffusion_run_configs::Union{Vector{Diffusion_run_config}, Nothing} = nothing
end

"""
    ensemble(ensemble_config::Ensemble_config, get_metrics::Function)

Runs an ensemble of experiments with the given configuration and returns the results as a DataFrame.

# Arguments
- `ensemble_config::Ensemble_config`: The configuration to run the ensemble with.
- `get_metrics::Function`: The function to get the metrics from the model.

# Returns
- `dataframes::Vector{DataFrame}`: The results of the ensemble as a DataFrame.
"""
function ensemble(ensemble_config::Ensemble_config, get_metrics::Function)
	dataframes = Vector{DataFrame}()
	init_election = parse_data(ensemble_config.data_path)
	if ensemble_config.remove_candidate_ids !== nothing
		init_election = remove_candidates(init_election, ensemble_config.remove_candidate_ids)
	end

	# election
	for (i, sampling_config) in enumerate(ensemble_config.sampling_configs)
		election = sample(deepcopy(init_election), sampling_config)

		prev_configs = Dict()
		prev_configs["sampling_config"] = sampling_config
		# model
		for (j, voter_config) in enumerate(ensemble_config.voter_configs)
			voter_config = resolve_dependencies(voter_config, prev_configs)
			prev_configs["voter_config"] = voter_config

			voters = init_voters(election.votes, voter_config)

			for (k, graph_config) in enumerate(ensemble_config.graph_configs)
				graph_config = resolve_dependencies(graph_config, prev_configs)
				prev_configs["graph_config"] = graph_config

				social_network = init_graph(voters, graph_config)
				model = General_model(voters, social_network, election.candidates)

				accumulator = Accumulator(get_metrics)
				add_metrics!(accumulator, model)

				# no diffusion
				if ensemble_config.diffusion_run_configs === nothing
					experiment_config = Experiment_config(
						election_config=Election_config(data_path=ensemble_config.data_path, remove_candidate_ids=ensemble_config.remove_candidate_ids, sampling_config=sampling_config),
						model_config=General_model_config(voter_config=voter_config, graph_config=graph_config),
						diffusion_config=nothing
					)
					df = hcat(DataFrame("experiment_config" => experiment_config), accumulated_metrics(accumulator))
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

					for (m, diffusion_run_config) in enumerate(ensemble_config.diffusion_run_configs)
						for n in eachindex(diffusion_run_config.mutation_configs)
							diffusion_run_config.mutation_configs[n] = resolve_dependencies(diffusion_run_config.mutation_configs[n], prev_configs)
						end
						prev_configs["diffusion_run_config"] = diffusion_run_config

						accumulator_diffusion = deepcopy(accumulator)
						model_diffusion = deepcopy(model_init)
						run!(model_diffusion, diffusion_run_config; accumulator=accumulator_diffusion)

						experiment_config = Experiment_config(
							election_config=Election_config(data_path=ensemble_config.data_path, remove_candidate_ids=ensemble_config.remove_candidate_ids, sampling_config=sampling_config),
							model_config=General_model_config(voter_config=voter_config, graph_config=graph_config),
							diffusion_config=Diffusion_config(diffusion_init_config=diffusion_init_config, diffusion_run_config=diffusion_run_config)
						)
						experiment_configs = fill(experiment_config, diffusion_run_config.diffusion_steps + 1)

						df = hcat(DataFrame("experiment_config" => experiment_configs, "diffusion_step" => collect(0:diffusion_run_config.diffusion_steps)), accumulated_metrics(accumulator_diffusion))
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

"""
    resolve_dependencies(config::Abstract_config, prev_configs)

Resolves the dependencies of the given config.

# Arguments
- `config::Abstract_config`: The config to resolve the dependencies of.
- `prev_configs::Dict`: The previous configs.

# Returns
- `config::Abstract_config`: The config with resolved dependencies.
"""
function resolve_dependencies(config::Abstract_config, prev_configs)
	return config
end

"""
    save_ensemble(ensemble_config::Ensemble_config, dataframe::DataFrame, path::String)

Saves the ensemble to the given path.

# Arguments
- `ensemble_config::Ensemble_config`: The ensemble config.
- `dataframe::DataFrame`: The ensemble results.
- `path::String`: The path to save the ensemble to.
"""
function save_ensemble(ensemble_config::Ensemble_config, dataframe::DataFrame, path::String)
	try
		jldsave(path; ensemble_config, dataframe)
	catch
		mkpath(dirname(path))
		jldsave(path; ensemble_config, dataframe)
	end
end

"""
    load_ensemble(path::String)

Loads the ensemble from the given path.

# Arguments
- `path::String`: The path to load the ensemble from.

# Returns
- `ensemble_config::Ensemble_config`: The ensemble config.
- `dataframe::DataFrame`: The ensemble results.
"""
function load_ensemble(path::String)
	return load(path, "ensemble_config"), load(path, "dataframe")
end
