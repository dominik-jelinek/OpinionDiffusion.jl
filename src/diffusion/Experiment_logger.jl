struct Experiment_logger
	experiment_dir::String
	diffusion_step::Vector{Int64}
	interval::Int64
end

"""
	Experiment_logger(; log_dir::String="./logs", experiment_name::String="experiment", interval::Int64=1)

Creates a new experiment logger.

# Arguments
- `log_dir::String="./logs"`: The directory to log the experiment to.
- `experiment_name::String="experiment"`: The name of the experiment.
- `interval::Int64=1`: The interval to log the model at.

# Returns
- `experiment_logger::Experiment_logger`: The experiment logger.
"""
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

"""
	trigger(experiment_logger::Experiment_logger, model::Abstract_model)

Triggers the experiment logger to log the given model.

# Arguments
- `experiment_logger::Experiment_logger`: The experiment logger.
- `model::Abstract_model`: The model to log.
"""
function trigger(experiment_logger::Experiment_logger, model::Abstract_model)
	if experiment_logger.diffusion_step[1] == 0
		throw(error("Experiment_logger has not been initialized, use init_experiment function"))
	end

	if experiment_logger.diffusion_step[1] % experiment_logger.interval == 0
		jldsave("$(experiment_logger.experiment_dir)/model_$(experiment_logger.diffusion_step[1]).jld2"; model)
	end

	experiment_logger.diffusion_step[1] += 1
end

"""
	load_model(experiment_dir::String, diffusion_step::Int64=-1)

Loads the model from the given experiment_dir and diffusion_step.

# Arguments
- `experiment_dir::String`: The directory of the experiment.
- `diffusion_step::Int64=-1`: The diffusion step to load the model from.

# Returns
- `model::Abstract_model`: The model.
"""
function load_model(experiment_dir::String, diffusion_step::Int64)
	if diffusion_step == -1
		diffusion_step = last_log_idx(experiment_dir)
	end

	return load_model("$(experiment_dir)/model_$(diffusion_step).jld2")
end

"""
	load_model(path::String)

Loads the model from the given path.

# Arguments
- `path::String`: The path to load the model from.

# Returns
- `model::Abstract_model`: The model.
"""
function load_model(path::String)
	return load(path, "model")
end

"""
	last_log_idx(experiment_dir::String)

Returns the last diffusion step logged in the given experiment_dir.

# Arguments
- `experiment_dir::String`: The directory of the experiment.

# Returns
# - `idx::Int64`: The last diffusion step logged.
"""
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
