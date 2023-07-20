@kwdef struct Diffusion_config <: Abstract_config
	diffusion_init_configs::Vector{T} where {T<:Abstract_mutation_init_config}
	diffusion_run_config::Diffusion_run_config
end

@kwdef struct Action
	operation::String
	ID::Union{Int64,Tuple{Int64,Int64}}
end

"""
	diffusion(model::Abstract_model, diffusion_config::Diffusion_config)::Tuple{Abstract_model, Vector{Vector{Action}}}

Diffuses the given model with the given diffusion_config.

# Arguments
- `model::Abstract_model`: The model to diffuse.
- `diffusion_config::Diffusion_config`: The config to diffuse the model with.

# Returns
- `model::Abstract_model`: The diffused model.
- `actions::Vector{Vector{Action}}`: The actions taken during diffusion.
"""
function diffusion(model, diffusion_config::Diffusion_config; accumulator=nothing, experiment_logger=nothing)
	model = deepcopy(model)
	diffusion_config = deepcopy(diffusion_config)

	actions = diffusion!(model, diffusion_config; accumulator=accumulator, experiment_logger=experiment_logger)

	return model, actions
end

"""
	diffusion!(model::Abstract_model, diffusion_config::Diffusion_config)::Vector{Vector{Action}}

Diffuses the given model with the given diffusion_config.

# Arguments
- `model::Abstract_model`: The model to diffuse.
- `diffusion_config::Diffusion_config`: The config to diffuse the model with.

# Returns
- `actions::Vector{Vector{Action}}`: The actions taken during diffusion.
"""
function diffusion!(model, diffusion_config::Diffusion_config; accumulator=nothing, experiment_logger=nothing)
	init_diffusion!(model, diffusion_config.diffusion_init_configs)
	return run_diffusion!(model, diffusion_config.diffusion_run_config; accumulator=accumulator, experiment_logger=experiment_logger)
end

"""
	run_diffusion(model::Abstract_model, diffusion_config::Diffusion_config)::Tuple{Vector{Vector{Action}}, Accumulator}

Diffuses the given model with the given diffusion_config.

# Arguments
- `model::Abstract_model`: The model to diffuse.
- `diffusion_config::Diffusion_config`: The config to diffuse the model with.

# Returns
- `actions::Vector{Vector{Action}}`: The actions taken during diffusion.
- `accumulator::Accumulator`: The accumulator containing the accumulated data.
"""
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

"""
	run_diffusion!(model::Abstract_model, diffusion_init_configs::Vector{Abstract_mutation_init_config})

Diffuses the given model with the given diffusion_init_configs.

# Arguments
- `model::Abstract_model`: The model to diffuse.
- `diffusion_init_configs::Vector{Abstract_mutation_init_config}`: The configs to diffuse the model with.

# Returns
- `actions::Vector{Vector{Action}}`: The actions taken during diffusion.
"""
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

"""
	_run_diffusion!(model::Abstract_model, mutation_configs::Vector{Abstract_mutation_config})

Diffuses the given model with the given mutation_configs.

# Arguments
- `model::Abstract_model`: The model to diffuse.
- `mutation_configs::Vector{Abstract_mutation_config}`: The configs to diffuse the model with.

# Returns
- `actions::Vector{Action}`: The actions taken during diffusion.
"""
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
