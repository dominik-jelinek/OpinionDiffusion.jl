function init_diffusion(
	model::T,
	diff_init_configs::Vector{U}
	) where {T<:Abstract_model, U<:Abstract_diff_init_config}

	model_cp = deepcopy(model)
	init_diffusion!(model_cp, diff_init_configs)

	return model_cp
end

function init_diffusion!(
	model::T,
	diff_init_configs::Vector{U}
) where {T<:Abstract_model, U<:Abstract_diff_init_config}

	for diff_init_config in diff_init_configs
		init_diffusion!(model, diff_init_config)
	end
end

function init_diffusion!(
	model::T,
	diff_init_config::U
) where {U<:Abstract_diff_init_config}

	throw(NotImplementedError("init_diffusion!"))
end

@kwdef struct Diffusion_config
	diffusion_steps::Int64
	diffusion_configs::Vector{Abstract_diff_config}
end

function get_rng(diff_config::T) where {T<:Abstract_diff_config}
	return diff_config.rng
end

function run(
	model::T,
	diffusion_config::Diffusion_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
) where {T<:Abstract_model}

	model = deepcopy(model)
	diff_configs = deepcopy(diffusion_config)
	experiment_logger = deepcopy(experiment_logger)
	accumulator = deepcopy(accumulator)

	actions = run!(model, diffusion_config; experiment_logger=experiment_logger, accumulator=accumulator)

	return model, actions, accumulator
end

function run!(
	model::T,
	diffusion_config::Diffusion_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
) where {T<:Abstract_model}
	actions = Vector{Vector{Action}}()

	for _ in 1:diff_configs.diffusion_steps
		push!(actions, _run!(model, diffusion_config.diffusion_configs; experiment_logger=experiment_logger, accumulator=accumulator))
	end

	return actions
end

function _run!(
	model::T,
	diff_configs::Vector{Abstract_diff_config};
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
) where {T<:Abstract_model}

	actions = diffusion!(model, diff_configs)

	if accumulator !== nothing
		add_metrics!(accumulator, get_metrics(model))
	end

	if experiment_logger !== nothing
		trigger(model, experiment_logger)
	end

	return actions
end

function diffusion!(model::T, diffusion_configs::Vector{<:Abstract_diff_config}) where {T<:Abstract_model}
	actions = Vector{Action}()

	for diffusion_config in diffusion_configs
		append!(actions, diffusion!(model, diffusion_config))
	end

	return actions
end

function diffusion!(model, diffusion_configs::T) where {T<:Abstract_diff_config}
	throw(NotImplementedError("diffusion!"))
end

@kwdef struct Ensemble_config
	input_filename::String
	selection_configs::Vector{Union{Selection_config, Nothing}}

	voter_init_configs::Vector{Abstract_voter_config}
	graph_init_configs::Vector{Abstract_graph_config}

	diff_init_configs::Vector{Union{Vector{Abstract_diff_init_config}, Nothing}}
	diff_configs::Vector{Union{Vector{Abstract_diff_config}, Nothing}}
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
@kwdef struct Experiment_config
	input_filename::String
	selection_config::Union{Selection_config, Nothing}

	voter_init_config::Abstract_voter_config
	graph_init_config::Abstract_graph_config

	diff_init_config::Union{Vector{Abstract_diff_init_config}, Nothing}
	diff_config::Union{Vector{Abstract_diff_config}, Nothing}
end

function ensemble(init_election::Election, ensemble_config::Ensemble_config, get_metrics::Function)
	dataframes = Vector{DataFrame}()

	# election
	for (i, selection_config) in enumerate(ensemble_config.selection_configs)
		election = select(init_election, selection_config)

		prev_configs = Dict()
		prev_configs["selection_config"] = selection_config
		# model
		for (j, voter_init_config) in enumerate(ensemble_config.voter_init_configs)
			voter_init_config = resolve_dependencies(voter_init_config, prev_configs)
			prev_configs["voter_init_config"] = voter_init_config

			voters = init_voters(election.votes, voter_init_config)

			for (k, graph_init_config) in enumerate(ensemble_config.graph_init_configs)
				graph_init_config = resolve_dependencies(graph_init_config, prev_configs)
				prev_configs["graph_init_config"] = graph_init_config

				social_network = init_graph(voters, graph_init_config)
				model = General_model(voters, social_network, election.party_names, election.candidates)

				accumulator = init_accumulator(get_metrics)
				add_metrics!(accumulator, model)

				# no diffusion
				if ensemble_config.diffusions == 0 || length(ensemble_config.diff_configs) == 0
					df = hcat(Dataframe(prev_configs), accumulated_metrics(diff_accumulator))
					df.diffusion_step = [0]
					push!(dataframes, df)
					continue
				end

				# diffusion
				for (l, diff_init_config) in enumerate(ensemble_config.diff_init_configs)
					for n in eachindex(diff_init_config)
						diff_init_config[n] = resolve_dependencies(diff_init_config[n], prev_configs)
					end
					prev_configs["diff_init_config"] = diff_init_config

					model_init = deepcopy(model)
					init_diffusion!(model_init, diff_init_config)

					for (m, diff_config) in enumerate(ensemble_config.diff_configs)
						for n in eachindex(diff_config)
							diff_config[n] = resolve_dependencies(diff_config[n], prev_configs)
						end
						prev_configs["diff_config"] = diff_config

						diff_accumulator = deepcopy(accumulator)
						model_diff = deepcopy(model_init)
						run!(model_diff, ensemble_config.diffusions, diff_config; accumulator=diff_accumulator)

						expanded_configs = Dict(key => fill(value, ensemble_config.diffusions + 1) for (key, value) in prev_configs)

						df = hcat(Dataframe(expanded_configs), accumulated_metrics(diff_accumulator))
						df.diffusion_step = collect(0:ensemble_config.diffusions)

						push!(dataframes, df)

						delete!(prev_configs, "diff_config")
					end

					delete!(prev_configs, "diff_init_config")
				end

				delete!(prev_configs, "graph_init_config")
			end

			delete!(prev_configs, "voter_init_config")
		end
	end

	return vcat(dataframes...)
end

function resolve_dependencies(config::T, prev_configs) where {T<:Abstract_config}
	return config
end