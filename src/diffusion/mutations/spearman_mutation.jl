@kwdef struct SP_mutation_init_config <: Abstract_mutation_init_config
	rng_seed::UInt32
	stubbornness_distr::Distributions.UnivariateDistribution
end

"""
	init_mutation!(model::T, mutation_init_config::SP_mutation_init_config)

Initializes the given model with the given mutation_init_config.

# Arguments
- `model::T`: The model to initialize.
- `mutation_init_config::SP_mutation_init_config`: The config to initialize the model with.

# Returns
- `model::T`: The initialized model.
"""
function init_mutation!(model::T, mutation_init_config::SP_mutation_init_config) where {T<:Abstract_model}
	rng = Random.MersenneTwister(mutation_init_config.rng_seed)
	voters = get_voters(model)
	set_property!(voters, "stubbornness", rand(rng, mutation_init_config.stubbornness_distr, length(voters)))
end

@kwdef struct SP_mutation_config <: Abstract_mutation_config
	rng::Random.MersenneTwister
	evolve_vertices::Float64
	attract_proba::Float64
	change_rate::Float64
	normalize_shifts::Bool
end

"""
	mutate!(model::T, mutation_config::SP_mutation_config)

Mutates the given model with the given mutation_config.

# Arguments
- `model::T`: The model to mutate.
- `mutation_config::SP_mutation_config`: The config to mutate the model with.

# Returns
- `actions::Vector{Action}`: The actions taken during mutation.
"""
function mutate!(model::T, mutation_config::SP_mutation_config) where {T<:Abstract_model}
	rng = mutation_config.rng
	voters = get_voters(model)
	actions = Vector{Action}()
	evolve_vertices = mutation_config.evolve_vertices

	sample_size = ceil(Int, evolve_vertices * length(voters))
	vertex_ids = StatsBase.sample(rng, 1:length(voters), sample_size, replace=true)

	for id in vertex_ids
		neighbor = select_neighbor(voters[id], model; rng=rng)
		if neighbor === nothing
			continue
		end

		append!(actions, average_all!(voters[id], neighbor, mutation_config.attract_proba, mutation_config.change_rate, mutation_config.normalize_shifts; rng=rng))
	end

	return actions
end

"""
	average_all!(self::Spearman_voter, neighbor::Spearman_voter, attract_proba, change_rate, normalize=nothing; rng=Random.GLOBAL_RNG)

Average the opinions of the given self and neighbor voter. The opinions are averaged with the given attract_proba and change_rate. If normalize is given, the shifts are normalized.

# Arguments
- `self::Spearman_voter`: The first voter.
- `neighbor::Spearman_voter`: The second voter.
- `attract_proba::Float64`: The probability to attract.
- `change_rate::Float64`: The rate of change.
- `normalize::Tuple{Bool, Bool}`: Whether to normalize the shifts. The first element is whether to normalize the shifts of the first voter, the second element is whether to normalize the shifts of the second voter.

# Returns
- `actions::Vector{Action}`: The actions taken during averaging.
"""
function average_all!(self::Spearman_voter, neighbor::Spearman_voter, attract_proba, change_rate, normalize=nothing; rng=Random.GLOBAL_RNG)
	opinion_1 = get_opinion(self)
	opinion_2 = get_opinion(neighbor)

	shifts_1 = (opinion_2 - opinion_1) / 2
	shifts_2 = shifts_1 .* (-1.0)

	method = "attract"
	if rand(rng) > attract_proba
		#repel
		method = "repel"
		shifts_1, shifts_2 = shifts_2, shifts_1
	end

	if normalize !== nothing && normalize[1]
		shifts_1 = normalize_shifts(shifts_1, opinion_1, self.weights)
		shifts_2 = normalize_shifts(shifts_2, opinion_2, neighbor.weights)
	end

	cp_1 = deepcopy(self)
	cp_2 = deepcopy(neighbor)
	opinion_1 .+= shifts_1 * (1.0 - get_property(self, "stubbornness")) * change_rate
	#opinion_2 .+= shifts_2 * (1.0 - get_property(neighbor, "stubbornness")) * change_rate
	return [Action(method, (get_ID(neighbor), get_ID(self)))]#, Action(method, (get_ID(self), get_ID(neighbor)), cp_2, deepcopy(neighbor))]
end

"""
	normalize_shifts(shifts::Vector{Float64}, opinion::Vector{Float64}, weights::Vector{Float64})

Normalizes the given shifts with the given opinion and weights.

# Arguments
- `shifts::Vector{Float64}`: The shifts to normalize.
- `opinion::Vector{Float64}`: The opinion to normalize with.
- `weights::Vector{Float64}`: The weights to normalize with.

# Returns
- `normalized::Vector{Float64}`: The normalized shifts.
"""
function normalize_shifts(shifts::Vector{Float64}, opinion::Vector{Float64}, weights::Vector{Float64})
	min_opin, max_opin = weights[1], weights[end]
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

"""
	normalize_shift(shift::Float64, can_opinion::Float64, min_opin, max_opin)

Normalizes the given shift with the given opinion and weights.

# Arguments
- `shift::Float64`: The shift to normalize.
- `can_opinion::Float64`: The opinion to normalize with.
- `min_opin::Float64`: The minimum opinion.
- `max_opin::Float64`: The maximum opinion.

# Returns
- `normalized::Float64`: The normalized shift.
"""
function normalize_shift(shift::Float64, can_opinion::Float64, min_opin, max_opin)
	if shift == 0.0 || min_opin <= can_opinion || can_opinion <= max_opin
		return shift
	end

	return shift * (sign(shift) == 1.0 ? 2^(-can_opinion + max_opin) : 2^(can_opinion - min_opin))
end
