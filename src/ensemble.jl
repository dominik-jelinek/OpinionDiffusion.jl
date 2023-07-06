@kwdef struct Ensemble_config
	input_filename::String
	remove_candidates_ids::Union{Vector{Vector{Int64}}, Nothing} = nothing
	sampling_configs::Vector{Union{Sampling_config, Nothing}} = [nothing]

	voter_configs::Vector{Abstract_voter_config}
	graph_configs::Vector{Abstract_graph_config}

	diffusion_init_configs::Union{Vector{Vector{Abstract_mutation_init_config}}, Nothing} = nothing
	diffusion_run_configs::Union{Vector{Diffusion_run_config}, Nothing} = nothing
end

function Base.length(ensemble_config::Ensemble_config)
	len = 1
	if length(ensemble_config.sampling_configs) > 0
		len *= length(ensemble_config.sampling_configs)
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

function ensemble(ensemble_config::Ensemble_config, get_metrics::Function)
	dataframes = Vector{DataFrame}()
	init_election = parse_data(ensemble_config.input_filename)
	if ensemble_config.remove_candidates_ids !== nothing
		init_election = remove_candidates(election, ensemble_config.remove_candidates_ids)
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
				model = General_model(voters, social_network, election.party_names, election.candidates)

				accumulator = Accumulator(get_metrics)
				add_metrics!(accumulator, model)

				# no diffusion
				if ensemble_config.diffusion_configs === nothing
					experiment_config = Experiment_config(
						election_config=Election_config(data_path=ensemble_config.input_filename, remove_candidates_ids=ensemble_config.remove_candidates_ids, sampling_config=sampling_config),
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
							diffusion_run_config.mutation_configs[n] = resolve_dependencies(diffusion_config.mutation_configs[n], prev_configs)
						end
						prev_configs["diffusion_run_config"] = diffusion_run_config

						accumulator_diffusion = deepcopy(accumulator)
						model_diffusion = deepcopy(model_init)
						run!(model_diffusion, diffusion_run_config; accumulator=accumulator_diffusion)

						experiment_config = Experiment_config(
							election_config=Election_config(data_path=ensemble_config.input_filename, remove_candidates_ids=ensemble_config.remove_candidates_ids, sampling_config=sampling_config),
							model_config=General_model_config(voter_config=voter_config, graph_config=graph_config),
							diffusion_config=Diffusion_config(diffusion_init_config=diffusion_init_config, diffusion_run_config=diffusion_run_config)
						)
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
	election_config::Election_config
	model_config::Abstract_model_config
	diffusion_config::Union{Diffusion_config, Nothing} = nothing
end

function run_experiment(config::Experiment_config; experiment_name="experiment", get_metrics::Function=nothing, checkpoint::Int=1)
	# Election
	election = init_election(config.election_config)

	# Model
	model = init_model(election, config.model_config)

	# Diffusion
	accumulator = nothing
	if get_metrics !== nothing
		accumulator = Accumulator(get_metrics)
		add_metrics!(accumulator, model)
	end

	experiment_logger = nothing
	if checkpoint > 0
		model_logger = Model_logger(election, model, config.model_config)
		experiment_logger = Experiment_logger(model_logger, diffusion_config, experiment_name=experiment_name, checkpoint=checkpoint)
	end

	if config.diffusion_config !== nothing
		run_diffusion!(model, config.diffusion_config; accumulator=accumulator, experiment_logger=experiment_logger)
	end

	return get_metrics !== nothing ? accumulated_metrics(accumulator) : nothing
end

function resolve_dependencies(config::Abstract_config, prev_configs)
	return config
end

function save_ensemble(ensemble_config::Ensemble_config, dataframe::DataFrame, path::String)
	try
		jldsave(path; ensemble_config, dataframe)
	catch
		mkpath(dirname(path))
		jldsave(path; ensemble_config, dataframe)
	end
end

function load_ensemble(path::String)
	return load(path, "ensemble_config"), load(path, "dataframe")
end