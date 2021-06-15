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
struct Experiment
    model::Spearman_model
    parties::Vector{String}
    candidates::Vector{Candidate}

    sampled_voter_ids::Vector{Int64}
    diffusion_metrics::Spearman_metrics

    exp_dir::String
    diff_counter::Vector{Int64}
end

function Experiment(model, parties, candidates, exp_config)
    exp_dir = "$(model.log_dir)/experiment_$(model.exp_counter[1])"
    mkpath(exp_dir)
    YAML.write_file("$(exp_dir)/exp_config.yml", exp_config)
    diff_counter = [1]
    model.exp_counter[1] += 1

    if exp_config["reduce_dim_config"]["used"]
        mkpath(exp_dir * "/images")
    end

    sampled_voter_ids = 1:length(model.voters)
    sampled_opinions = model.voters
    if exp_config["sample_size"] != 0
        sampled_voter_ids = sample(1:length(model.voters), exp_config["sample_size"], replace=false)
        sampled_opinions = reduce(hcat, [voter.opinion for voter in model.voters[sampled_voter_ids]])
    end
    jldsave("$(exp_dir)/sampled_voter_ids.jld2"; sampled_voter_ids)

    metrics = Spearman_metrics(model.voters, model.social_network, length(candidates))

    #visualize_voters(projections, clusters, expConfig["reduceDimConfig"]["method"], expConfig["clusteringConfig"]["method"], expDir, counter)
    #visualizeVoters(model.voters([sampled_voter_ids]), candidates, parties, expConfig, expDir * "/images", 0)

    return Experiment(model, parties, candidates, sampled_voter_ids, metrics, exp_dir, diff_counter)
end

function run_experiment!(experiment, diffusion_config)
    for i in 1:diffusion_config["diffusions"]
        diffusion!(experiment.model, diffusion_config)
        
        update_metrics!(experiment.model)
        
        #cluster_labels, clusters = clustering(sampled_opinions, candidates, parties, exp_config["clustering_config"])
        #projections = reduceDim(sampled_opinions, expConfig["reduceDimConfig"])
        #if expConfig["reduce_dim_config"]["used"]
        #    visualizeVoters(sampled_opinions, sampled_voters, candidates, parties, exp_config, exp_dir * "/images", diff_counter)      
        #end

        if i % diffusion_config["checkpoint"]
            log(experiment.model)
        end
        experiment.diff_counter[1] += 1
    end

    jldsave("$(exp_dir)/diffusion_metrics.jld2"; experiment.diffusion_metrics)
    return experiment.diffusion_metrics
end


function Spearman_metrics(voters, social_network, can_count)
    dict = degree_histogram(social_network)
    keyss = collect(keys(dict))
    votes = get_votes(voters)
    
    return Spearman_metrics([minimum(keyss)], 
                            [ne(social_network) * 2 / nv(social_network)], 
                            [maximum(keyss)], 
                            [plurality_voting(votes, can_count, true)], 
                            [borda_voting(votes, can_count, true)], 
                            [copeland_voting(votes, can_count)])
end

function update_metrics!(model::Spearman_model)
    dict = degree_histogram(model.social_network)
    keyss = collect(keys(dict))
    push!(metrics.min_degrees, minimum(keyss))
    push!(metrics.avg_degrees, ne(model.social_network) * 2 / nv(model.social_network))
    push!(metrics.max_degrees, maximum(keyss))
    
    votes = get_votes(model.voters)
    push!(metrics.plurality_votings, plurality_voting(votes, can_count, true))
    push!(metrics.borda_votings, borda_voting(votes, can_count, true))
    push!(metrics.copeland_votings, copeland_voting(votes, can_count))
 end

 