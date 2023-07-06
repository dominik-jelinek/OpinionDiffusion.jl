struct Model_logger
	model_dir::String
end

function Model_logger(
	election::Election,
	model::Abstract_model,
	model_configs::Abstract_model_config;
	log_dir::String="./logs",
	model_name::String="model"
)

	model_dir = "$(log_dir)/$(model_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	mkpath(model_dir)

	save_election(election, "$(model_dir)/election.jld2")
	save_model(model, "$(model_dir)/model.jld2")
	save_config(model_configs, "$(model_dir)/model_configs.jld2")

	return Model_logger(model_dir)
end

function load_election(model_logger::Model_logger)
	return load_election("$(model_logger.model_dir)/election.jld2")
end

function load_model(model_logger::Model_logger)
	return load_model("$(model_logger.model_dir)/model.jld2")
end

function load_configs(model_logger::Model_logger)
	return load_config("$(model_logger.model_dir)/model_configs.jld2")
end