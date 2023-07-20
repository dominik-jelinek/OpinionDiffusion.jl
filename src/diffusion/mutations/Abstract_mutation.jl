"""
	init_diffusion(model::Abstract_model, diffusion_init_configs::Vector{Abstract_mutation_init_config})

Initializes the given model with the given diffusion_init_configs.

# Arguments
- `model::Abstract_model`: The model to initialize.
- `diffusion_init_configs::Vector{Abstract_mutation_init_config}`: The configs to initialize the model with.

# Returns
- `model_cp::Abstract_model`: The initialized model.
"""
function init_diffusion(
	model::Abstract_model,
	mutation_init_configs::Vector{Abstract_mutation_init_config}
)
	model_cp = deepcopy(model)
	init_diffusion!(model_cp, mutation_init_configs)

	return model_cp
end

"""
	init_diffusion!(model::Abstract_model, diffusion_init_configs::Vector{Abstract_mutation_init_config})

Initializes the given model with the given diffusion_init_configs.

# Arguments
- `model::Abstract_model`: The model to initialize.
- `diffusion_init_configs::Vector{Abstract_mutation_init_config}`: The configs to initialize the model with.

# Returns
- `model::Abstract_model`: The initialized model.
"""
function init_diffusion!(
	model::Abstract_model,
	mutation_init_configs::Vector{T} where {T<:Abstract_mutation_init_config}
)
	for mutation_init_config in mutation_init_configs
		init_mutation!(model, mutation_init_config)
	end
end

"""
	run_diffusion(model::Abstract_model, diffusion_run_config::Diffusion_run_config)::Vector{Vector{Action}}

Diffuses the given model with the given diffusion_run_config.

# Arguments
- `model::Abstract_model`: The model to diffuse.
- `diffusion_run_config::Diffusion_run_config`: The config to diffuse the model with.

# Returns
- `actions::Vector{Vector{Action}}`: The actions taken during diffusion.
"""
function init_mutation!(
	model::Abstract_model,
	mutation_init_configs::Vector{T} where {T<:Abstract_mutation_init_config}
)
	throw(NotImplementedError("init_mutation!"))
end

@kwdef struct Diffusion_run_config
	diffusion_steps::Int64
	mutation_configs::Vector{Abstract_mutation_config}
end

"""
	run_mutations(model::Abstract_model, mutation_configs::Vector{Abstract_mutation_config})::Vector{Action}

Runs the given mutations on the given model.

# Arguments
- `model::Abstract_model`: The model to run the mutations on.
- `mutation_configs::Vector{Abstract_mutation_config}`: The mutations to run on the model.

# Returns
- `actions::Vector{Action}`: The actions taken during the mutations.
"""
function run_mutations!(
	model::Abstract_model,
	mutation_configs::Vector{Abstract_mutation_config}
)
	actions = Vector{Action}()

	for mutation_config in mutation_configs
		append!(actions, mutate!(model, mutation_config))
	end

	return actions
end

"""
	mutate!(model::Abstract_model, mutation_config::Abstract_mutation_config)::Vector{Action}

Mutates the given model with the given mutation_config.

# Arguments
- `model::Abstract_model`: The model to mutate.
- `mutation_config::Abstract_mutation_config`: The config to mutate the model with.

# Returns
- `actions::Vector{Action}`: The actions taken during mutation.
"""
function mutate!(
	model::Abstract_model,
	mutation_config::Abstract_mutation_config
)
	throw(NotImplementedError("mutate!"))
end
