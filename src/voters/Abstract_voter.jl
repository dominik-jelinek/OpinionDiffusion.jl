abstract type Abstract_voter end

function get_opinion(voter::Abstract_voter)
    return voter.opinion
end