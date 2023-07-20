#=
@kwdef struct Graph_mutation_init_config <: Abstract_mutation_init_config
	rng_seed::UInt32
	openmindedness_distr::Distributions.UnivariateDistribution
end

function init_diffusion!(model::Abstract_model, mutation_init_config::Graph_mutation_init_config)
	rng = Random.MersenneTwister(mutation_init_config.rng_seed)
	voters = get_voters(model)
	set_property!(voters, "openmindedness", rand(rng, mutation_init_config.openmindedness_distr, length(voters)))
end
=#

@kwdef struct Graph_mutation_config <: Abstract_mutation_config
	rng::Random.MersenneTwister
	evolve_edges::Float64
	homophily::Float64
end

"""
	mutate!(model::Abstract_model, mutation_config::Graph_mutation_config)

Mutates the given model with the given mutation_config.

# Arguments
- `model::Abstract_model`: The model to mutate.
- `mutation_config::Graph_mutation_config`: The config to mutate the model with.

# Returns
- `actions::Vector{Action}`: The actions taken during mutation.
"""
function mutate!(model::Abstract_model, mutation_config::Graph_mutation_config)
	voters = get_voters(model)
	actions = Vector{Action}()
	evolve_vertices = mutation_config.evolve_edges
	rng = mutation_config.rng

	sample_size = ceil(Int, evolve_vertices * length(voters))
	vertex_ids = StatsBase.sample(rng, 1:length(voters), sample_size, replace=true)

	for id in vertex_ids
		append!(actions, edge_diffusion!(voters[id], model, homophily; rng=rng))
	end

	return actions
end

"""
	edge_diffusion!(self::Voter, model::Abstract_model, homophily::Float64)

Diffuses the given voter's edges according to the given homophily.

# Arguments
- `self::Voter`: The voter to diffuse.
- `model::Abstract_model`: The model to diffuse the voter in.
- `homophily::Float64`: The homophily to diffuse the voter with.
- `rng::Random.MersenneTwister`: The random number generator to use.

# Returns
- `actions::Vector{Action}`: The actions taken during diffusion.
"""
function edge_diffusion!(self, model, homophily; rng=Random.GLOBAL_RNG)
	voters, social_network = get_voters(model), get_social_network(model)
	ID = self.ID
	distances = get_distance(self, voters)

	neibrs = neighbors(social_network, ID)
	if length(neibrs) == 0
		# do not add or remove edges because this would change average degree
		return []
	end

	#remove one neighboring edge
	degree_probs = 1 ./ degree(social_network, neibrs)
	distance_probs = 2 .^ distances[neibrs]
	probs = homophily .* degree_probs ./ sum(degree_probs) + (1.0 - homophily) .* distance_probs ./ sum(distance_probs)

	to_remove = neibrs[StatsBase.sample(rng, 1:length(neibrs), StatsBase.Weights(probs))]
	rem_edge!(social_network, ID, to_remove)

	#add edge
	degree_probs = degree(social_network)
	distance_probs = (1 / 2) .^ distances
	probs = homophily .* degree_probs ./ sum(degree_probs) + (1.0 - homophily) .* distance_probs ./ sum(distance_probs)
	probs[ID] = 0.0
	probs[neighbors(social_network, ID)] .= 0.0

	to_add = StatsBase.sample(rng, 1:length(voters), StatsBase.Weights(probs))
	add_edge!(social_network, ID, to_add)

	return [Action("remove_edge", (ID, to_remove)), Action("add_edge", (ID, to_add))]
end
