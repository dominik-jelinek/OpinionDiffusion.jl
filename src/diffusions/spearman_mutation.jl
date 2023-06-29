@kwdef struct SP_mutation_init_config <: Abstract_mutation_init_config
	rng_seed::UInt32
	stubbornness_distr::Distributions.UnivariateDistribution
end

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
		shifts_1 = normalize_shifts(shifts_1, opinion_1, voter_1.weights)
		shifts_2 = normalize_shifts(shifts_2, opinion_2, voter_2.weights)
	end

	cp_1 = deepcopy(voter_1)
	cp_2 = deepcopy(voter_2)
	opinion_1 .+= shifts_1 * (1.0 - get_property(voter_1, "stubbornness")) * change_rate
	opinion_2 .+= shifts_2 * (1.0 - get_property(voter_2, "stubbornness")) * change_rate
	return [Action(method, (get_ID(voter_2), get_ID(voter_1)), cp_1, deepcopy(voter_1)), Action(method, (get_ID(voter_1), get_ID(voter_2)), cp_2, deepcopy(voter_2))]
end

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

function normalize_shift(shift::Float64, can_opinion::Float64, min_opin, max_opin)
	if shift == 0.0 || min_opin <= can_opinion || can_opinion <= max_opin
		return shift
	end

	return shift * (sign(shift) == 1.0 ? 2^(-can_opinion + max_opin) : 2^(can_opinion - min_opin))
end