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

struct Experiment
    model::Spearman_model

    sampled_voter_ids::Vector{Int64}
    diffusion_metrics::Spearman_metrics

    voter_visualization_config
    exp_dir::String
    diff_counter::Vector{Int64}
end

function Experiment(model, candidates, exp_config)
    if model.exp_counter != 1
        reset_model!(model)
    end
    exp_dir = "$(model.log_dir)/experiment_$(model.exp_counter[1])"
    mkpath(exp_dir)
    YAML.write_file("$(exp_dir)/exp_config.yml", exp_config)
    jldsave("$(exp_dir)/model_0.jld2"; model)
    
    diff_counter = [1]
    model.exp_counter[1] += 1

    if exp_config.voter_visualization_config.used
        mkpath(exp_dir * "/images")
    end

    sampled_voter_ids = Nothing
    if exp_config.sample_size != 0
        sampled_voter_ids = StatsBase.sample(1:length(model.voters), exp_config.sample_size, replace=false)
        jldsave("$(exp_dir)/sampled_voter_ids.jld2"; sampled_voter_ids)
    end
    
    metrics = Spearman_metrics(model, candidates, sampled_voter_ids, exp_config.voter_visualization_config)
    
    return Experiment(model, sampled_voter_ids, metrics, exp_config.voter_visualization_config, exp_dir, diff_counter)
end

function run_experiment!(experiment, candidates, diffusion_config)
    changes = Vector{Vector{Float64}}()
    for i in 1:diffusion_config.diffusions
        prev_opinions = get_opinions(experiment.model.voters)
        diffusion!(experiment.model, diffusion_config)
        opinion_change = get_opinions(experiment.model.voters) - prev_opinions
        
        #normalized_change = mapslices(normalize, opinion_change; dims=1)*2 .- 1.0
        #push!(changes, vec(sum(normalized_change, dims=2)))
        push!(changes, vec(sum(sign.(opinion_change), dims=2)))

        update_metrics!(experiment, candidates)

        if i % diffusion_config.checkpoint == 0
            log(experiment)
        end
        experiment.diff_counter[1] += 1
    end

    jldsave("$(experiment.exp_dir)/diffusion_metrics.jld2"; experiment.diffusion_metrics)

    return experiment.diffusion_metrics, changes
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

function update_metrics!(experiment::Experiment, candidates)
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

function log(experiment::Experiment)
    jldsave("$(experiment.exp_dir)/model_$(experiment.diff_counter[1]).jld2"; experiment.model)
end