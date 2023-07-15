struct Experiment_logger
	experiment_dir::String
	diffusion_step::Vector{Int64}
	interval::Int64
end

function Experiment_logger(
	;
	log_dir::String = "./logs",
	experiment_name::String = "experiment",
	interval::Int64 = 1
)
	experiment_dir = "$(log_dir)/$(experiment_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	mkpath(experiment_dir)

	return Experiment_logger(experiment_dir, [0], interval)
end

function trigger(experiment_logger::Experiment_logger, model::Abstract_model)
	if experiment_logger.diffusion_step[1] == 0
		throw(error("Experiment_logger has not been initialized, use init_experiment function"))
	end

	if experiment_logger.diffusion_step[1] % experiment_logger.interval == 0
		jldsave("$(experiment_logger.experiment_dir)/model_$(experiment_logger.diffusion_step[1]).jld2"; model)
	end

	experiment_logger.diffusion_step[1] += 1
end

function load_model(experiment_dir::String, diffusion_step::Int64)
	if diffusion_step == -1
		diffusion_step = last_log_idx(experiment_dir)
	end

	return load_model("$(experiment_dir)/model_$(diffusion_step).jld2")
end

function load_model(path::String)
	return load(path, "model")
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