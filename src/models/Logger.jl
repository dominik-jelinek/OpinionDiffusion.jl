# Logger takes model and creates log based on provided name 
struct Logger
    model_dir::String
    diff_counter::Union{Vector{Int64}, Nothing}
    exp_dir::Union{String, Nothing}
    checkpoint::Union{Int64, Nothing}
end

# new
function Logger(model::Abstract_model, model_configs; log_dir::String="./logs" , model_name::String="model")

    model_dir = "$(log_dir)/$(model_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(model_dir)

    jldsave("$(model_dir)/model.jld2"; model)
    jldsave("$(model_dir)/model_configs.jld2"; model_configs)

    return Logger(model_dir, nothing, nothing, nothing)
end

# run
function log(logger::Logger, model::T) where {T<:Abstract_model}
    if logger.diff_counter[1] % logger.checkpoint == 0
        save_model(logger, model)
    end

    logger.diff_counter[1] += 1
end

function init_experiment(logger::Logger, diffusion_configs, exp_name::String, checkpoint::Int64)
    exp_dir = "$(logger.model_dir)/$(exp_name)_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(exp_dir)

    jldsave("$(exp_dir)/diffusion_configs.jld2"; diffusion_configs)

    return Logger(logger.model_dir, [0], exp_dir, checkpoint)
end

function save_model(logger::Logger, model::T) where {T<:Abstract_model}
    jldsave("$(logger.exp_dir)/model_$(logger.diff_counter[1]).jld2"; model)
end

function load_model(model_dir::String)
    return load("$(model_dir)/model.jld2", "model")
end

function load_model(exp_dir::String, idx::Int64)
    if idx == -1
        idx = last_log_idx(exp_dir)
    end

    return load("$(exp_dir)/model_$(idx).jld2", "model")
end

function load_models(exp_dir::String, start_idx::Int64, end_idx::Int64)
    if end_idx == -1
        end_idx = last_log_idx(exp_dir)
    end
    models = Vector{Abstract_model}(undef, end_idx - start_idx + 1)

    for (i, j) in enumerate(start_idx:end_idx)
        models[i] = load_model(exp_dir, j)
    end

    return models
end

function last_log_idx(exp_dir)
    idx = -1
    for filename in readdir(exp_dir)
        val = parse(Int64, chop(split(filename, "_")[end], tail=5))
        if val > idx
            idx = val
        end
    end

    return idx
end