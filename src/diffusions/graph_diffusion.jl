@kwdef struct Graph_init_diff_config <: Abstract_init_diff_config
    openmindednesses::Vector{Float64}
end

@kwdef struct Graph_diff_config <: Abstract_diff_config
    evolve_edges::Float64
    homophily::Float64
end

function init_diffusion!(model::T, init_diff_config::Graph_init_diff_config; rng=Random.GLOBAL_RNG) where T <: Abstract_model
    set_property!(get_voters(model), "openmindedness", init_diff_config.openmindednesses)
end

function diffusion!(model::T, diffusion_config::Graph_diff_config; rng=Random.GLOBAL_RNG) where T <: Abstract_model
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
        
        append!(actions, step!(voters[id], neighbor, voter_diff_config.attract_proba, length(get_candidates(model)); rng=Random.GLOBAL_RNG))
    end

    return actions
end

function edge_diffusion!(self, model, popularity_ratio; rng=Random.Global)
    voters, social_network = get_voters(model), get_social_network(model)
    ID = self.ID
    distances = get_distance(self, voters)

    neibrs = neighbors(social_network, ID)
    if length(neibrs) == 0
        # do not add or remove edges because this would change average degree
        return
    end

    #remove one neighboring edge
    degree_probs = 1 ./ degree(social_network, neibrs)
    distance_probs = 2 .^ distances[neibrs]
    probs = popularity_ratio .* degree_probs ./ sum(degree_probs) + (1.0 - popularity_ratio) .* distance_probs ./ sum(distance_probs)

    to_remove = StatsBase.sample(rng, 1:length(neibrs), StatsBase.Weights(probs))
    rem_edge!(social_network, ID, neibrs[to_remove])

    #add edge
    degree_probs = degree(social_network)
    distance_probs = (1 / 2) .^ distances
    probs = popularity_ratio .* degree_probs ./ sum(degree_probs) + (1.0 - popularity_ratio) .* distance_probs ./ sum(distance_probs)
    probs[ID] = 0.0
    probs[neighbors(social_network, ID)] .= 0.0

    to_add = StatsBase.sample(rng, 1:length(voters), StatsBase.Weights(probs))
    add_edge!(social_network, ID, to_add)
end