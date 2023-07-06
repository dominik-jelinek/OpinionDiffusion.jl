@kwdef struct Diffusion_run_config
	diffusion_steps::Int64
	mutation_configs::Vector{Abstract_mutation_config}
end

@kwdef struct Diffusion_config <: Abstract_config
	diffusion_init_config::Union{Vector{Abstract_mutation_init_config}, Nothing}
	diffusion_run_config::Diffusion_run_config
end

function run_diffusion!(model, diffusion_config::Diffusion_config; accumulator=nothing, experiment_logger=nothing)
	init_diffusion!(model, diffusion_config.diffusion_init_config)
	run!(model, diffusion_config.diffusion_run_config; accumulator=accumulator, experiment_logger=experiment_logger)

	return accumulated_metrics(accumulator)
end

function init_diffusion(
	model::Abstract_model,
	mutation_init_configs::Vector{Abstract_mutation_init_config}
)
	model_cp = deepcopy(model)
	init_diffusion!(model_cp, mutation_init_configs)

	return model_cp
end

function init_diffusion!(
	model::Abstract_model,
	mutation_init_configs::Vector{Abstract_mutation_init_config}
)
	for mutation_init_config in mutation_init_configs
		init_mutation!(model, mutation_init_config)
	end
end

function init_mutation!(
	model::Abstract_model,
	mutation_init_configs::Vector{Abstract_mutation_init_config}
)
	throw(NotImplementedError("init_mutation!"))
end

function run(
	model::Abstract_model,
	diffusion_config::Diffusion_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)

	model = deepcopy(model)
	diffusion_config = deepcopy(diffusion_config)
	experiment_logger = deepcopy(experiment_logger)
	accumulator = deepcopy(accumulator)

	actions = run!(model, diffusion_config; experiment_logger=experiment_logger, accumulator=accumulator)

	return actions, accumulator
end

function run!(
	model::Abstract_model,
	diffusion_run_config::Diffusion_run_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)
	actions = Vector{Vector{Action}}()

	for _ in 1:diffusion_run_config.diffusion_steps
		push!(actions, _run!(model, diffusion_run_config.mutation_configs; experiment_logger=experiment_logger, accumulator=accumulator))
	end

	return actions
end

function _run!(
	model::Abstract_model,
	mutation_configs::Vector{Abstract_mutation_config};
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)

	actions = _run_diffusion!(model, mutation_configs)

	if accumulator !== nothing
		add_metrics!(accumulator, model)
	end

	if experiment_logger !== nothing
		trigger(model, experiment_logger)
	end

	return actions
end

function _run_diffusion!(model::Abstract_model, mutation_configs::Vector{Abstract_mutation_config})
	actions = Vector{Action}()

	for mutation_config in mutation_configs
		append!(actions, mutate!(model, mutation_config))
	end

	return actions
end

function mutate!(model::Abstract_model , mutation_config::Abstract_mutation_config)
	throw(NotImplementedError("mutate!"))
end