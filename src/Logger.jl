struct Logger
    model_dir::String
    exp_dir::String
    diff_counter::Vector{Int64}
end

#new
function Logger(model)
    model_dir = "logs/model_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(model_dir)
    exp_dir = "$(model_dir)/experiment_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(exp_dir)
    
    save_log(model, model_dir)
    save_log(model, exp_dir, 0)
    diff_counter = [0]

    return Logger(model, model_dir, exp_dir, diff_counter)
end

#restart
function Logger(model, model_dir)
    exp_dir = "$(model_dir)/experiment_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(exp_dir)
    
    save_log(model, exp_dir, 0)    
    diff_counter = [0]

    return Logger(model, model_dir, exp_dir, diff_counter)
end

#load
function Logger(model, model_dir, exp_dir, idx::Int64)
    if idx == -1
        idx = last_log_idx(exp_dir)
    else
        #remove logs of models that were created after loaded log 
        for file in readdir(exp_dir)
            if parse(Int64, chop(split(file, "_")[end], tail=5)) > idx
                rm(exp_dir * "/" * file)
            end
        end
    end

    return Logger(model, model_dir, exp_dir, [idx])
end

function save_log(logger::Logger, model)
    save_log(model, logger.exp_dir, logger.diff_counter[1])
end

function load_log(logger::Logger)
    return load_log(logger.exp_dir, logger.diff_counter[1])
end