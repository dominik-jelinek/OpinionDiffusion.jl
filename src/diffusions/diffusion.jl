function init_diffusion!(model, diffusion_configs::Vector{<:Abstract_init_diff_config}; rng=Random.GLOBAL_RNG)
    for diffusion_config in diffusion_configs
        init_diffusion!(model, diffusion_config; rng=rng)
    end
end

function init_diffusion!(model, diffusion_config::T; rng=Random.GLOBAL_RNG) where T <: Abstract_init_diff_config
    throw(NotImplementedError("init_diffusion!"))
end

function diffusion!(model::T, diffusion_configs::Vector{<:Abstract_diff_config}; rng=Random.GLOBAL_RNG) where T <: Abstract_model
    actions = Vector{Action}()
    
    for diffusion_config in diffusion_configs
        append!(actions, diffusion!(model, diffusion_config; rng=rng))
    end

    return actions
end

function diffusion!(model, diffusion_configs::T; rng=Random.GLOBAL_RNG) where T <: Abstract_diff_config
    throw(NotImplementedError("diffusion!"))
end