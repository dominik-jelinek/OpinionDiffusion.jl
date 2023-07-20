@kwdef struct Experiment_config
	election_config::Election_config
	model_config::Abstract_model_config
	diffusion_config::Union{Diffusion_config, Nothing} = nothing
end

"""
    init_experiment(experiment_logger::Experiment_logger, model::Abstract_model, config::Experiment_config)

Initializes the experiment with the given experiment_logger, model, and configuration.

# Arguments
- `experiment_logger::Experiment_logger`: The experiment_logger to initialize the experiment with.
- `model::Abstract_model`: The model to initialize the experiment with.
- `config::Experiment_config`: The configuration to initialize the experiment with.
"""
function init_experiment(
	experiment_logger::Experiment_logger,
	model::Abstract_model,
	config::Experiment_config
)
	if experiment_logger.diffusion_step[1] != 0
		error("Experiment_logger has already been initialized")
	end
	experiment_dir = experiment_logger.experiment_dir

	jldsave("$(experiment_logger.experiment_dir)/model_0.jld2"; model)
	experiment_logger.diffusion_step[1] = 1
	jldsave("$(experiment_dir)/experiment_config.jld2"; config)
end

"""
    load_config(experiment_dir::String)

Loads the configuration from the given experiment_dir.

# Arguments
- `experiment_dir::String`: The directory of the experiment.

# Returns
- `config::Experiment_config`: The configuration.
"""
function load_config(experiment_dir::String)
	return load(experiment_dir * "/experiment_config.jld2", "config")
end

"""
    run_experiment(config::Experiment_config)

Runs the experiment with the given configuration.

# Arguments
- `config::Experiment_config`: The configuration to run the experiment with.
- `experiment_logger::Union{Experiment_logger, Nothing}=nothing`: The experiment logger to log the experiment with.
- `accumulator::Union{Accumulator, Nothing}=nothing`: The accumulator to accumulate the experiment with.
"""
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
	if accumulator !== nothing
		add_metrics!(accumulator, model)
	end

	if experiment_logger !== nothing
		init_experiment(experiment_logger, model, config)
	end

	if config.diffusion_config !== nothing
		diffusion!(model, config.diffusion_config; accumulator=accumulator, experiment_logger=experiment_logger)
	end
end
