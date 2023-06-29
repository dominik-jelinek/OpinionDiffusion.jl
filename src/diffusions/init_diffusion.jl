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