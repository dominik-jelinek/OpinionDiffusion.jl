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
end
struct Visualizations{Backend <: AbstractBackend}
    voter_visualizations::Vector{Plots.Plot{Backend}}
    degree_distributions::Vector{Plots.Plot{Backend}}
end

struct Experiment{Backend <: AbstractBackend} 
    model::Spearman_model

    sampled_voter_ids::Vector{Int64}
    diffusion_metrics::Spearman_metrics
    visualizations::Visualizations{Backend}

    voter_visualization_config
    exp_dir::String
    diff_counter::Vector{Int64}
end

function Experiment(model, candidates, parties, backend::Type{Backend}, exp_config) where Backend <: AbstractBackend
    exp_dir = "$(model.log_dir)/experiment_$(model.exp_counter[1])"
    mkpath(exp_dir)
    YAML.write_file("$(exp_dir)/exp_config.yml", exp_config)
    jldsave("$(exp_dir)/model_0.jld2"; model)
    
    diff_counter = [1]
    model.exp_counter[1] += 1

    if exp_config["voter_visualization_config"]["used"]
        mkpath(exp_dir * "/images")
    end

    sampled_voter_ids = Nothing
    if exp_config["sample_size"] != 0
        sampled_voter_ids = sample(1:length(model.voters), exp_config["sample_size"], replace=false)
        jldsave("$(exp_dir)/sampled_voter_ids.jld2"; sampled_voter_ids)
    end
    
    metrics = Spearman_metrics(model, length(candidates))
    visualizations = Visualizations(model, sampled_voter_ids, candidates, parties,  exp_dir, [0], exp_config["voter_visualization_config"])

    return Experiment{backend}(model, sampled_voter_ids, metrics, visualizations, exp_config["voter_visualization_config"], exp_dir, diff_counter)
end

function run_experiment!(experiment, candidates, parties, diffusion_config)
    for i in 1:diffusion_config["diffusions"]
        diffusion!(experiment.model, diffusion_config)
        
        update_metrics!(experiment, length(candidates))
        update_visualizations!(experiment, candidates, parties)
        #cluster_labels, clusters = clustering(sampled_opinions, candidates, parties, exp_config["clustering_config"])
        #projections = reduceDim(sampled_opinions, expConfig["reduceDimConfig"])
        #if expConfig["reduce_dim_config"]["used"]
        #    visualizeVoters(sampled_opinions, sampled_voters, candidates, parties, exp_config, exp_dir * "/images", diff_counter)      
        #end
        #visualize_voters(projections, clusters, expConfig["reduceDimConfig"]["method"], expConfig["clusteringConfig"]["method"], expDir, counter)
        #visualizeVoters(model.voters([sampled_voter_ids]), candidates, parties, expConfig, expDir * "/images", 0)

        if i % diffusion_config["checkpoint"] == 0
            log(experiment)
        end
        experiment.diff_counter[1] += 1
    end

    jldsave("$(experiment.exp_dir)/diffusion_metrics.jld2"; experiment.diffusion_metrics)
end


function Spearman_metrics(model::Spearman_model, can_count)
    dict = degree_histogram(model.social_network)
    keyss = collect(keys(dict))
    votes = get_votes(model.voters)

    return Spearman_metrics([minimum(keyss)], 
                            [ne(model.social_network) * 2 / nv(model.social_network)], 
                            [maximum(keyss)], 
                            [plurality_voting(votes, can_count, true)], 
                            [borda_voting(votes, can_count, true)], 
                            [copeland_voting(votes, can_count)])
end

function Visualizations(model, sampled_voter_ids, candidates, parties, exp_dir::String, diff_counter, voter_visualization_config)

    return Visualizations(  [draw_voter_visualization(model.voters, sampled_voter_ids, candidates, parties, exp_dir::String, diff_counter, voter_visualization_config)], 
                            [draw_degree_distribution(degree_histogram(model.social_network), exp_dir, diff_counter)]
                            )
end

function update_visualizations!(experiment::Experiment, candidates, parties)
    push!(experiment.visualizations.voter_visualizations, draw_voter_visualization(experiment.model.voters, experiment.sampled_voter_ids, candidates, parties, experiment.exp_dir, experiment.diff_counter, experiment.voter_visualization_config))
    push!(experiment.visualizations.degree_distributions, draw_degree_distribution(degree_histogram(experiment.model.social_network), experiment.exp_dir, experiment.diff_counter))
end

function update_metrics!(experiment::Experiment, can_count)
    social_network = experiment.model.social_network
    diffusion_metrics = experiment.diffusion_metrics

    dict = degree_histogram(social_network)
    keyss = collect(keys(dict))
    push!(diffusion_metrics.min_degrees, minimum(keyss))
    push!(diffusion_metrics.avg_degrees, ne(social_network) * 2 / nv(social_network))
    push!(diffusion_metrics.max_degrees, maximum(keyss))
    
    votes = get_votes(experiment.model.voters)
    push!(diffusion_metrics.plurality_votings, plurality_voting(votes, can_count, true))
    push!(diffusion_metrics.borda_votings, borda_voting(votes, can_count, true))
    push!(diffusion_metrics.copeland_votings, copeland_voting(votes, can_count))
 end

function log(experiment::Experiment)
    jldsave("$(experiment.exp_dir)/model_$(experiment.diff_counter[1]).jld2"; experiment.model)
end