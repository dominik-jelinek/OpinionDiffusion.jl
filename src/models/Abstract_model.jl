get_voters(model::T) where {T<:Abstract_model} = model.voters
get_social_network(model::T) where {T<:Abstract_model} = model.social_network
get_candidates(model::T) where {T<:Abstract_model} = model.candidates

#=
function run_ensemble_model(
    ensemble_size,
    diffusions,
    election,
    candidates,
    init_metrics,
    update_metrics!,
    model_config::Abstract_model_config,
    init_diff_config,
    diff_configs,
    log=false
)

    ens_metrics = Vector{Any}(undef, ensemble_size)

    @threads for i in 1:ensemble_size
        model_seed = rand(UInt32)
        model_rng = MersenneTwister(model_seed)
        model = init_model(election, candidates, model_config; rng=model_rng)
        metrics = init_metrics(model)

        init_diffusion!(model, init_diff_config; rng=model_rng)

        diffusion_seed = rand(UInt32)
        rng = MersenneTwister(diffusion_seed)

        actions = run!(model, diff_configs, diffusions; metrics=metrics, (update_metrics!)=update_metrics!, rng=rng)

        frequent_votes = get_frequent_votes(get_votes(get_voters(model)), 10)
        ens_metrics[i] = Dict("model_seed" => model_seed,
            "diffusion_seed" => diffusion_seed,
            "metrics" => metrics,
            "frequent_votes" => frequent_votes)
    end

    if log
        save_ensemble(model_config, diff_init_configs, diff_configs, ens_metrics)
    end

    return ens_metrics
end

function run_ensemble(model::Abstract_model, ensemble_size, diffusions, init_metrics, update_metrics!, diff_configs, logger=nothing)
    ens_metrics = Vector{Any}(undef, ensemble_size)

    init_diffusion!(model, diff_init_configs; rng=model_rng)

    @threads for i in 1:ensemble_size
        model_cp = deepcopy(model)
        metrics = deepcopy(init_metrics)

        diffusion_seed = rand(UInt32)
        rng = MersenneTwister(diffusion_seed)

        actions = run!(model_cp, diff_configs, diffusions; metrics=metrics, (update_metrics!)=update_metrics!, rng=rng)

        frequent_votes = get_frequent_votes(get_votes(get_voters(model_cp)), 10)
        ens_metrics[i] = Dict("diffusion_seed" => diffusion_seed,
            "metrics" => metrics,
            "frequent_votes" => frequent_votes)
    end

    if logger !== nothing
        save_ensemble(logger.model_dir, diff_configs, ens_metrics)
    end

    return ens_metrics
end
=#
function run(model::T, diff_configs, diffusions; logger=nothing, checkpoint=1, accumulator=nothing, get_metrics=nothing) where {T<:Abstract_model}
    model = deepcopy(model)
    actions = Vector{Vector{Action}}()
    
    for j in 1:diffusions
        push!(actions, _run!(model, diff_configs; logger=logger, checkpoint=checkpoint, accumulator=accumulator, get_metrics=get_metrics))
    end

    return model, actions
end

function run!(model::T, diff_configs, diffusions; logger=nothing, checkpoint=1, accumulator=nothing, get_metrics=nothing) where {T<:Abstract_model}
    diff_configs = deepcopy(diff_configs)
    actions = Vector{Vector{Action}}()
    
    for j in 1:diffusions
        push!(actions, _run!(model, diff_configs; logger=logger, checkpoint=checkpoint, accumulator=accumulator, get_metrics=get_metrics))
    end

    return actions
end

function _run!(model::T, diff_configs::Vector{Abstract_diff_config}; logger=nothing, checkpoint=1, accumulator=nothing, get_metrics=nothing) where {T<:Abstract_model}
    actions = diffusion!(model, diff_configs)

    if get_metrics !== nothing
        add_metrics!(accumulator, get_metrics(model))
    end

    if logger !== nothing
        logger.diff_counter[1] += 1

        if logger.diff_counter[1] % checkpoint == 0
            save_log(logger, model)
        end
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

function ensemble(ensemble_config::Ensemble_config, get_metrics)
    dataframes = Vector{DataFrame}()

    # election
    for (i, selection_config) in enumerate(ensemble_config.selection_configs)
        election = parse_data(ensemble_config.input_filename)
        election = select(election, selection_config)

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
                accumulator = init_accumulator(get_metrics(model))
                
                # no diffusion
                if ensemble_config.diffusions == 0 || length(ensemble_config.diff_configs) == 0
                    push!(dataframes, DataFrame(merge(prev_configs, accumulator)))
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
                        run!(model_diff, diff_config, ensemble_config.diffusions; accumulator=diff_accumulator, get_metrics=get_metrics)

                        expanded_configs = Dict(key => fill(value, ensemble_config.diffusions + 1) for (key, value) in prev_configs)

                        df = DataFrame(merge(expanded_configs, diff_accumulator))
                        df.diffusion_step = collect(1:ensemble_config.diffusions + 1)
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

    return dataframes
end