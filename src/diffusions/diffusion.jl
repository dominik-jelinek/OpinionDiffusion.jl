@kwdef struct Diffusion_config
	diffusion_steps::Int64
	mutation_configs::Vector{Abstract_mutation_config}
end

function get_rng(mutation_config::T) where {T<:Abstract_mutation_config}
	return mutation_config.rng
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

	return model, actions, accumulator
end

function run!(
	model::Abstract_model,
	diffusion_config::Diffusion_config;
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)
	actions = Vector{Vector{Action}}()

	for _ in 1:diffusion_config.diffusion_steps
		push!(actions, _run!(model, diffusion_config.mutation_configs; experiment_logger=experiment_logger, accumulator=accumulator))
	end

	return actions
end

function _run!(
	model::Abstract_model,
	mutation_configs::Vector{Abstract_mutation_config};
	experiment_logger::Union{Experiment_logger, Nothing}=nothing,
	accumulator::Union{Accumulator, Nothing}=nothing,
)

	actions = diffusion!(model, mutation_configs)

	if accumulator !== nothing
		add_metrics!(accumulator, get_metrics(model))
	end

	if experiment_logger !== nothing
		trigger(model, experiment_logger)
	end

	return actions
end

function diffusion!(model::Abstract_model, mutation_configs::Vector{Abstract_mutation_config})
	actions = Vector{Action}()

	for mutation_config in mutation_configs
		append!(actions, mutate!(model, mutation_config))
	end

	return actions
end

function mutate!(model::Abstract_model , mutation_config::Abstract_mutation_config)
	throw(NotImplementedError("mutate!"))
end