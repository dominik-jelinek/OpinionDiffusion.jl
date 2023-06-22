struct General_model <: Abstract_model
    voters::Vector{Abstract_voter}

    social_network::AbstractGraph

    party_names::Vector{String}
    candidates::Vector{Candidate}
end
