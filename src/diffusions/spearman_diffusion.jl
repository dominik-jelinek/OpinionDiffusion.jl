@kwdef struct SP_init_diff_config <: Abstract_init_diff_config
    stubbornnesses::Vector{Float64}
end

@kwdef struct SP_diff_config <: Abstract_diff_config
    evolve_vertices::Float64
    attract_proba::Float64
	change_rate::Float64
    normalize_shifts::Union{Nothing, Tuple{Bool, Float64, Float64}}
end

function init_diffusion!(model::T, init_diff_config::SP_init_diff_config; rng=Random.GLOBAL_RNG) where T <: Abstract_model
    set_property!(get_voters(model), "stubbornness", init_diff_config.stubbornnesses)
end

function diffusion!(model::T, diffusion_config::SP_diff_config; rng=Random.GLOBAL_RNG) where T <: Abstract_model
    voters = get_voters(model)
    actions = Vector{Action}()
    evolve_vertices = diffusion_config.evolve_vertices

    sample_size = ceil(Int, evolve_vertices * length(voters))
    vertex_ids = StatsBase.sample(rng, 1:length(voters), sample_size, replace=true)
    
    for id in vertex_ids
        neighbor = select_neighbor(voters[id], model; rng=rng)
        if neighbor === nothing
            continue
        end
        
        append!(actions, average_all!(voters[id], neighbor, diffusion_config.attract_proba, diffusion_config.change_rate, diffusion_config.normalize_shifts; rng=rng))
    end

    return actions
end

function average_all!(voter_1::Spearman_voter, voter_2::Spearman_voter, attract_proba, change_rate, normalize=nothing; rng=Random.GLOBAL_RNG)
    opinion_1 = get_opinion(voter_1)
    opinion_2 = get_opinion(voter_2)
    
    shifts_1 = (opinion_2 - opinion_1) / 2
    shifts_2 = shifts_1 .* (-1.0)
    
    method = "attract"
    if rand(rng) > attract_proba
        #repel
        method = "repel"
        shifts_1, shifts_2 = shifts_2, shifts_1
    end

    if normalize !== nothing && normalize[1]
        shifts_1 = normalize_shifts(shifts_1, opinion_1, normalize[2], normalize[3])
        shifts_2 = normalize_shifts(shifts_2, opinion_2, normalize[2], normalize[3])
    end

    cp_1 = deepcopy(voter_1)
    cp_2 = deepcopy(voter_2)
    opinion_1 .+= shifts_1 * (1.0 - get_property(voter_1, "stubbornness")) * change_rate
    opinion_2 .+= shifts_2 * (1.0 - get_property(voter_2, "stubbornness")) * change_rate
    return [Action(method, (get_ID(voter_2), get_ID(voter_1)), cp_1, deepcopy(voter_1)), Action(method, (get_ID(voter_1), get_ID(voter_2)), cp_2, deepcopy(voter_2))]
end

function normalize_shifts(shifts::Vector{Float64}, opinion::Vector{Float64}, min_opin, max_opin)
    # decrease opinion changes that push candidates outside of [min_opin, max_opin] boundary
    #safeguard
    if max_opin < min_opin
        min_opin, max_opin = max_opin, min_opin
    end

    normalized = Vector{Float64}(undef, length(shifts))
    for i in eachindex(shifts)
        normalized[i] = normalize_shift(shifts[i], opinion[i], min_opin, max_opin)
    end

    return normalized
end

function normalize_shift(shift::Float64, can_opinion::Float64, min_opin, max_opin)
    if shift == 0.0 || min_opin <= can_opinion || can_opinion <= max_opin
        return shift
    end  

    return shift * (sign(shift) == 1.0 ? 2^(-can_opinion + max_opin) : 2^(can_opinion - min_opin))
end