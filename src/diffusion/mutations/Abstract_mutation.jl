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

@kwdef struct Diffusion_run_config
	diffusion_steps::Int64
	mutation_configs::Vector{Abstract_mutation_config}
end

function run_mutations!(model::Abstract_model, mutation_configs::Vector{Abstract_mutation_config})
	actions = Vector{Action}()

	for mutation_config in mutation_configs
		append!(actions, mutate!(model, mutation_config))
	end

	return actions
end

function mutate!(
	model::Abstract_model,
	mutation_config::Abstract_mutation_config
)
	throw(NotImplementedError("mutate!"))
end