struct Spearman_metrics
    #min_distance::Vector{Int64}
    #avg_distance::Vector{Float64}
    #max_distance::Vector{Int64}
    
    #graph metrics
    min_degrees::Vector{Int}
    avg_degrees::Vector{Float64}
    max_degrees::Vector{Int}
 
    #election results
    plurality_votings::Vector{Vector{Float64}}
    borda_votings::Vector{Vector{Float64}}
    copeland_votings::Vector{Vector{Float64}}
    #STV

    projections::Vector{Matrix{Float64}}
    labels::Vector{Vector{Int64}} 
    clusters::Vector{Vector{Set{Int}}}
    degree_distributions::Vector{Dict{Int64, Int64}}
    edge_distances::Vector{Vector{Float64}}
end

struct Logger
    model::Spearman_model

    model_dir::String
    exp_dir::String
    diff_counter::Vector{Int64}
end

#new
function Logger(model)
    model_dir = "logs/model_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(model_dir)
    exp_dir = "$(model_dir)/experiment_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(exp_dir)
    
    save_log(model, model_dir)
    save_log(model, exp_dir, 0)
    diff_counter = [1]

    return Logger(model, model_dir, exp_dir, diff_counter)
end

#restart
function Logger(model, model_dir)
    exp_dir = "$(model_dir)/experiment_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath(exp_dir)
    
    save_log(model, exp_dir, 0)    
    diff_counter = [1]
    return Logger(model, model_dir, exp_dir, diff_counter)
end

#load
function Logger(model, model_dir, exp_dir, idx::Int64)
    #remove logs of models that were created after loaded log 
    for file in readdir(exp_dir)
        if parse(Int64, chop(split(file, "_")[end], tail=5)) > idx
            rm(exp_dir * "/" * file)
        end
    end

    return new(model, model_dir, exp_dir, [idx])
end

function run!(logger::Logger, diffusion_config)
    models = Vector{typeof(logger.model)}(undef, diffusion_config.diffusions)
    for i in 1:diffusion_config.diffusions
        diffusion!(logger.model, diffusion_config)
        models[i] = deepcopy(logger.model)
        
        save_log(logger)
        logger.diff_counter[1] += 1
    end

    return models
end

function run!(model::T, diffusion_config) where T<:Abstract_model
    models = Vector{T}(undef, diffusion_config.diffusions)
    for i in 1:diffusion_config.diffusions
        diffusion!(model, diffusion_config)
        models[i] = deepcopy(model)
    end

    return models
end

function Spearman_metrics(model, candidates, sampled_voter_ids, voter_visualization_config)
    dict = LightGraphs.degree_histogram(model.social_network)
    keyss = collect(keys(dict))
    votes = get_votes(model.voters)
    projections, labels, clusters = get_voter_vis(model.voters, sampled_voter_ids, candidates, voter_visualization_config)

    can_count = length(candidates)
    return Spearman_metrics([minimum(keyss)], 
                            [LightGraphs.ne(model.social_network) * 2 / LightGraphs.LightGraphs.nv(model.social_network)], 
                            [maximum(keyss)], 
                            [plurality_voting(votes, can_count, true)], 
                            [borda_voting(votes, can_count, true)], 
                            [copeland_voting(votes, can_count)],
                            [projections],
                            [labels],
                            [clusters], 
                            [LightGraphs.degree_histogram(model.social_network)],
                            [get_edge_distances(model.social_network, model.voters)])
end

function update_metrics!(experiment::Logger, candidates)
    social_network = experiment.model.social_network
    diffusion_metrics = experiment.diffusion_metrics

    dict = degree_histogram(social_network)
    keyss = collect(keys(dict))
    push!(diffusion_metrics.min_degrees, minimum(keyss))
    push!(diffusion_metrics.avg_degrees, LightGraphs.ne(social_network) * 2 / LightGraphs.nv(social_network))
    push!(diffusion_metrics.max_degrees, maximum(keyss))
    
    votes = get_votes(experiment.model.voters)
    can_count = length(candidates)
    push!(diffusion_metrics.plurality_votings, plurality_voting(votes, can_count, true))
    push!(diffusion_metrics.borda_votings, borda_voting(votes, can_count, true))
    push!(diffusion_metrics.copeland_votings, copeland_voting(votes, can_count))

    projections, labels, clusters = get_voter_vis(experiment.model.voters, experiment.sampled_voter_ids, candidates, experiment.voter_visualization_config)
    push!(diffusion_metrics.projections, projections)
    push!(diffusion_metrics.labels, labels)
    push!(diffusion_metrics.clusters, clusters)
    push!(diffusion_metrics.degree_distributions, LightGraphs.degree_histogram(experiment.model.social_network))
    push!(diffusion_metrics.edge_distances, get_edge_distances(experiment.model.social_network, experiment.model.voters))
end

function save_log(logger::Logger)
    save_log(logger.model, logger.exp_dir, logger.diff_counter[1])
end

function load_log(logger::Logger)
    return load("$(logger.exp_dir)/model_$(logger.diff_counter[1]).jld2", "model")
end