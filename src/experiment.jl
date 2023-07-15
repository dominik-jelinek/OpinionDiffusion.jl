function run_experiment(
	config::Experiment_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing
)
	# Election
	election = init_election(config.election_config)

	# Model
	model = init_model(election, config.model_config)

	# Diffusion
	if get_metrics !== nothing
		add_metrics!(accumulator, model)
	end

	if experiment_logger !== nothing
		init_experiment(experiment_logger, model, config)
	end

	if config.diffusion_config !== nothing
		run_diffusion!(model, config.diffusion_config; accumulator=accumulator, experiment_logger=experiment_logger)
	end
end