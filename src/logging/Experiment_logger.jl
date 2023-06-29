struct Experiment_logger
	experiment_dir::String
	diffusion_step::Vector{Int64}
	checkpoint::Int64
end

function Experiment_logger(
	model_logger::Model_logger,
	diffusion_configs;
	checkpoint::Int64 = 1,
	experiment_name::String = "experiment"
)
	experiment_dir = "$(model_logger.model_dir)/$(experiment_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	mkpath(experiment_dir)

	experiment_logger = Experiment_logger(experiment_dir, [0], checkpoint)

	save_model(model_logger, experiment_logger)
	save_configs(diffusion_configs, experiment_logger)

	return experiment_logger
end

function trigger(model::T, experiment_logger::Experiment_logger) where {T<:Abstract_model}
	experiment_logger.diffusion_step[1] += 1

	if experiment_logger.diffusion_step[1] % experiment_logger.checkpoint == 0
		save_model(model, experiment_logger)
	end
end

function save_model(model::T, experiment_logger::Experiment_logger) where {T<:Abstract_model}
	save_model(model, "$(experiment_logger.experiment_dir)/model_$(experiment_logger.diffusion_step[1]).jld2")
end

function load_models(experiment_dir::String, start_idx::Int64=0, end_idx::Int64=-1)
	if end_idx == -1
		end_idx = last_log_idx(experiment_dir)
	end
	models = Vector{Abstract_model}(undef, end_idx - start_idx + 1)

	for (i, j) in enumerate(start_idx:end_idx)
		models[i] = load_model(experiment_dir, j)
	end

	return models
end

function load_model(experiment_logger::Experiment_logger, idx::Int64=-1)
	if idx == -1
		idx = last_log_idx(experiment_logger.experiment_dir)
	end

	return load_model("$(experiment_logger.experiment_dir)/model_$(idx).jld2")
end

function last_log_idx(experiment_dir::String)
	idx = -1
	for filename in readdir(experiment_dir)
		val = parse(Int64, chop(split(filename, "_")[end], tail=5))
		if val > idx
			idx = val
		end
	end

	return idx
end

function save_configs(configs::T, experiment_logger::Experiment_logger) where {T<:Abstract_config}
	save_config(configs, "$(experiment_logger.experiment_dir)/diffusion_configs.jld2")
end

function load_configs(experiment_logger::Experiment_logger)
	return load_config("$(experiment_logger.experiment_dir)/diffusion_configs.jld2")
end