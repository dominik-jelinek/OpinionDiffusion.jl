struct Model_logger
	model_dir::String
end

function Model_logger(
	model::Abstract_model,
	model_configs;
	log_dir::String="./logs",
	model_name::String="model"
)

	model_dir = "$(log_dir)/$(model_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	mkpath(model_dir)

	model_logger = Model_logger(model_dir, [])
	save_model(model, model_logger)
	save_configs(model_configs, model_logger)

	return model_logger
end

function save_model(model::Abstract_model, model_logger::Model_logger)
	save_model(model, "$(model_logger.model_dir)/model.jld2")
end

function save_configs(model_configs, model_logger::Model_logger)
	save_config(model_configs, "$(model_logger.model_dir)/model_configs.jld2")
end

function load_model(model_logger::Model_logger)
	return load_model("$(model_logger.model_dir)/model.jld2")
end

function load_configs(model_logger::Model_logger)
	return load_config("$(model_logger.model_dir)/model_configs.jld2")
end