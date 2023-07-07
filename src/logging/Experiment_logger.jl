struct Experiment_logger
	experiment_dir::String
	diffusion_step::Vector{Int64}
	checkpoint::Int64
end

function Experiment_logger(
	model::Abstract_model,
	config::Experiment_config;
	checkpoint::Int64 = 1,
	experiment_name::String = "experiment",
	log_dir::String = "./logs"
)
	experiment_dir = "$(log_dir)/$(experiment_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	mkpath(experiment_dir)

	jldsave("$(experiment_dir)/model_0.jld2"; model)
	jldsave("$(experiment_dir)/experiment_config.jld2"; config)

	return Experiment_logger(experiment_dir, [1], checkpoint)
end

function trigger(model::T, experiment_logger::Experiment_logger) where {T<:Abstract_model}
	if experiment_logger.diffusion_step[1] % experiment_logger.checkpoint == 0
		jldsave("$(experiment_logger.experiment_dir)/model_$(experiment_logger.diffusion_step[1]).jld2"; model)
	end

	experiment_logger.diffusion_step[1] += 1
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

function load_model(experiment_dir::String, diffusion_step::Int64)
	if diffusion_step == -1
		idx = last_log_idx(experiment_dir)
	end

	return load_model("$(experiment_dir)/model_$(idx).jld2")
end

function load_model(path::String)
	return load(path, "model")
end

function load_config(path::String)
	return load(path, "config")
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