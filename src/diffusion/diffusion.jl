@kwdef struct Diffusion_config <: Abstract_config
	diffusion_init_configs::Vector{T} where {T<:Abstract_mutation_init_config}
	diffusion_run_config::Diffusion_run_config
end

@kwdef struct Action
	operation::String
	ID::Union{Int64,Tuple{Int64,Int64}}
end

function diffusion(model, diffusion_config::Diffusion_config; accumulator=nothing, experiment_logger=nothing)
	model = deepcopy(model)
	diffusion_config = deepcopy(diffusion_config)

	actions = diffusion!(model, diffusion_config; accumulator=accumulator, experiment_logger=experiment_logger)

	return model, actions
end

function diffusion!(model, diffusion_config::Diffusion_config; accumulator=nothing, experiment_logger=nothing)
	init_diffusion!(model, diffusion_config.diffusion_init_configs)
	return run_diffusion!(model, diffusion_config.diffusion_run_config; accumulator=accumulator, experiment_logger=experiment_logger)
end

function run_diffusion(
	model::Abstract_model,
	diffusion_config::Diffusion_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)

	model = deepcopy(model)
	diffusion_config = deepcopy(diffusion_config)
	experiment_logger = deepcopy(experiment_logger)
	accumulator = deepcopy(accumulator)

	actions = run_diffusion!(model, diffusion_config; experiment_logger=experiment_logger, accumulator=accumulator)

	return actions, accumulator
end

function run_diffusion!(
	model::Abstract_model,
	diffusion_run_config::Diffusion_run_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)
	actions = Vector{Vector{Action}}()

	for _ in 1:diffusion_run_config.diffusion_steps
		push!(actions, _run_diffusion!(model, diffusion_run_config.mutation_configs; experiment_logger=experiment_logger, accumulator=accumulator))
	end

	return actions
end

function _run_diffusion!(
	model::Abstract_model,
	mutation_configs::Vector{Abstract_mutation_config};
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)

	actions = run_mutations!(model, mutation_configs)

	if accumulator !== nothing
		add_metrics!(accumulator, model)
	end

	if experiment_logger !== nothing
		trigger(experiment_logger, model)
	end

	return actions
end