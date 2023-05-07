# Logger takes model and creates log based on provided name 
struct Logger
    model_dir::String
    exp_dir::String
    diff_counter::Vector{Int64}
end

#new
function Logger(model::Abstract_model, model_name="model", exp_name="experiment")
    model_dir = "logs/$(model_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(model_dir)
    exp_dir = "$(model_dir)/$(exp_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(exp_dir)
    
    println("Saving log")
    @time save_log(model, model_dir)
    save_log(model, exp_dir, 0)
    diff_counter = [0]

    return Logger(model_dir, exp_dir, diff_counter)
end

function load_model(model_dir::String, exp_name="experiment")
    exp_dir = "$(model_dir)/$(exp_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(exp_dir)
    
    model = load_log(model_dir)
    save_log(model, exp_dir, 0)    
    diff_counter = [0]

    return model, Logger(model_dir, exp_dir, diff_counter)
end

function load_model(model_dir::String, exp_dir::String, idx::Int64, overwrite::Bool, exp_name="experiment")
    last_log = false
    if idx == -1
        idx = last_log_idx(exp_dir)
        last_log = true
    end
    model = load_log(exp_dir, idx)
    diff_counter = [idx]

    if overwrite
        # continue in loaded model and delete all subsequent logs
        if last_log
            for file in readdir(exp_dir)
                if parse(Int64, chop(split(file, "_")[end], tail=5)) > idx
                    rm(exp_dir * "/" * file)
                end
            end
        end
    else
        # create new experiment with initial state of the loaded model
        exp_dir = "$(model_dir)/$(exp_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        mkpath(exp_dir)

        save_log(model, exp_dir, 0)    
        diff_counter = [0]
    end

    return model, Logger(model_dir, exp_dir, diff_counter)
end

function save_log(logger::Logger, model)
    save_log(model, logger.exp_dir, logger.diff_counter[1])
end

function load_log(logger::Logger)
    return load_log(logger.exp_dir, logger.diff_counter[1])
end

function save_ensemble(model_dir::String, diffusion_config, gathered_metrics)
	jldsave("$(model_dir)/ensemble_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).jld2"; diffusion_config, gathered_metrics)
end

function save_ensemble(model_config, diffusion_config, gathered_metrics)
	jldsave("logs/ensemble_model_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).jld2"; model_config, diffusion_config, gathered_metrics)
end