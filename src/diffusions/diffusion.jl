# rng mandatory
function get_rng(diffusion_config::T) where {T<:Abstract_diff_config}
    return diffusion_config.rng
end

function init_diffusion!(model, diff_init_configs::Vector{<:Abstract_diff_init_config})
    for diff_init_config in diff_init_configs
        init_diffusion!(model, diff_init_config)
    end
end

function init_diffusion!(model, diff_init_config::T) where {T<:Abstract_diff_init_config}
    throw(NotImplementedError("init_diffusion!"))
end

function diffusion!(model::T, diffusion_configs::Vector{<:Abstract_diff_config}) where {T<:Abstract_model}
    actions = Vector{Action}()

    for diffusion_config in diffusion_configs
        append!(actions, diffusion!(model, diffusion_config))
    end

    return actions
end

function diffusion!(model, diffusion_configs::T) where {T<:Abstract_diff_config}
    throw(NotImplementedError("diffusion!"))
end