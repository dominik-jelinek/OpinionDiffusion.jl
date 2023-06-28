get_voters(model::T) where {T<:Abstract_model} = model.voters
get_social_network(model::T) where {T<:Abstract_model} = model.social_network
get_candidates(model::T) where {T<:Abstract_model} = model.candidates

function run(
    model::T,
    diffusions::Int64,
    diff_configs::Vector{Abstract_diff_config};
    logger::Union{Logger, Nothing}=nothing,
    accumulator::Union{Accumulator, Nothing}=nothing,
) where {T<:Abstract_model}

    model = deepcopy(model)

    actions = run!(model, diff_configs, diffusions; logger=logger, accumulator=accumulator)

    return model, actions
end

function run!(
    model::T,
    diffusions::Int64,
    diff_configs::Vector{Abstract_diff_config}; 
    logger::Union{Logger, Nothing}=nothing,
    accumulator::Union{Accumulator, Nothing}=nothing,
) where {T<:Abstract_model}

    diff_configs = deepcopy(diff_configs)
    actions = Vector{Vector{Action}}()

    for j in 1:diffusions
        push!(actions, _run!(model, diff_configs; logger=logger, accumulator=accumulator))
    end

    return actions
end

function _run!(
    model::T, 
    diff_configs::Vector{Abstract_diff_config}; 
    logger::Union{Logger, Nothing}=nothing,
    accumulator::Union{Accumulator, Nothing}=nothing,
) where {T<:Abstract_model}

    actions = diffusion!(model, diff_configs)

    if get_metrics !== nothing
        add_metrics!(accumulator, get_metrics(model))
    end

    if logger !== nothing
        log(logger, model)
    end

    return actions
end

function save_log(model::T, model_dir) where {T<:Abstract_model}
    jldsave("$(model_dir)/model_init.jld2"; model)
end

function save_log(model::T, exp_dir::String, idx::Int64) where {T<:Abstract_model}
    jldsave("$(exp_dir)/model_$(idx).jld2"; model)
end

#restart
function load_log(model_dir::String)
    return load("$(model_dir)/model_init.jld2", "model")
end

#load
function load_log(exp_dir::String, idx::Int64)
    if idx == -1
        idx = last_log_idx(exp_dir)
    end

    return load("$(exp_dir)/model_$(idx).jld2", "model")
end

function load_log(exp_dir::String, model_name::String)
    return load("$(exp_dir)/$(model_name)", "model")
end

function load_logs(exp_dir::String, start_idx::Int64, end_idx::Int64)
    if end_idx == -1
        end_idx = last_log_idx(exp_dir)
    end
    models = Vector{Abstract_model}(undef, end_idx - start_idx + 1)

    for (i, j) in enumerate(start_idx:end_idx)
        models[i] = load_log(exp_dir, j)
    end

    return models
end

function select_neighbor(self, model; rng=Random.GLOBAL_RNG)
    voters = get_voters(model)
    social_network = get_social_network(model)
    neighbors_ = neighbors(social_network, get_ID(self))

    if length(neighbors_) == 0
        return nothing
    end

    neighbor_id = neighbors_[rand(rng, 1:end)]
    neighbor = voters[neighbor_id]

    return neighbor
end

@kwdef struct Ensemble_config
    input_filename::String
    selection_configs::Vector{Selection_config}

    voter_init_configs::Vector{Abstract_voter_init_config}
    graph_init_configs::Vector{Abstract_graph_init_config}

    diffusions::Int64
    diff_init_configs::Vector{Vector{Abstract_diff_init_config}}
    diff_configs::Vector{Vector{Abstract_diff_config}}
end

function Base.length(ensemble_config::Ensemble_config)
    len = length(ensemble_config.selection_configs) *
          length(ensemble_config.voter_init_configs) *
          length(ensemble_config.graph_init_configs)

    if length(ensemble_config.diff_init_configs) > 0
        len *= length(ensemble_config.diff_init_configs)
    end

    if length(ensemble_config.diff_configs) > 0
        len *= length(ensemble_config.diff_configs)
    end

    return len
end
@kwdef struct Experiment_config
    input_filename::String
    selection_config::Selection_config

    voter_init_config::Abstract_voter_init_config
    graph_init_config::Abstract_graph_init_config

    diffusions::Int64
    diff_init_config::Vector{Abstract_diff_init_config}
    diff_config::Vector{Abstract_diff_config}
end

function resolve_dependencies(config, prev_configs)
    return config
end

function ensemble(init_election, ensemble_config::Ensemble_config, get_metrics)
    dataframes = Vector{DataFrame}()

    # election
    for (i, selection_config) in enumerate(ensemble_config.selection_configs)
        election = select(init_election, selection_config)

        prev_configs = Dict()
        prev_configs["selection_config"] = selection_config
        # model
        for (j, voter_init_config) in enumerate(ensemble_config.voter_init_configs)
            voter_init_config = resolve_dependencies(voter_init_config, prev_configs)
            prev_configs["voter_init_config"] = voter_init_config

            voters = init_voters(election.votes, voter_init_config)

            for (k, graph_init_config) in enumerate(ensemble_config.graph_init_configs)
                graph_init_config = resolve_dependencies(graph_init_config, prev_configs)
                prev_configs["graph_init_config"] = graph_init_config

                social_network = init_graph(voters, graph_init_config)
                model = General_model(voters, social_network, election.party_names, election.candidates)
                accumulator = init_accumulator(get_metrics, model)

                # no diffusion
                if ensemble_config.diffusions == 0 || length(ensemble_config.diff_configs) == 0
                    df = hcat(Dataframe(prev_configs), accumulated_metrics(diff_accumulator))
                    df.diffusion_step = [0]
                    push!(dataframes, df)
                    continue
                end

                # diffusion
                for (l, diff_init_config) in enumerate(ensemble_config.diff_init_configs)
                    for n in eachindex(diff_init_config)
                        diff_init_config[n] = resolve_dependencies(diff_init_config[n], prev_configs)
                    end
                    prev_configs["diff_init_config"] = diff_init_config

                    model_init = deepcopy(model)
                    init_diffusion!(model_init, diff_init_config)

                    for (m, diff_config) in enumerate(ensemble_config.diff_configs)
                        for n in eachindex(diff_config)
                            diff_config[n] = resolve_dependencies(diff_config[n], prev_configs)
                        end
                        prev_configs["diff_config"] = diff_config

                        diff_accumulator = deepcopy(accumulator)
                        model_diff = deepcopy(model_init)
                        run!(model_diff, ensemble_config.diffusions, diff_config; accumulator=diff_accumulator)

                        expanded_configs = Dict(key => fill(value, ensemble_config.diffusions + 1) for (key, value) in prev_configs)

                        df = hcat(Dataframe(expanded_configs), accumulated_metrics(diff_accumulator))
                        df.diffusion_step = collect(0:ensemble_config.diffusions)

                        push!(dataframes, df)

                        delete!(prev_configs, "diff_config")
                    end

                    delete!(prev_configs, "diff_init_config")
                end

                delete!(prev_configs, "graph_init_config")
            end

            delete!(prev_configs, "voter_init_config")
        end
    end

    return vcat(dataframes...)
end