abstract type Abstract_voter end

function get_opinion(voter::Abstract_voter)
    return voter.opinion
end

function get_distance(voter1::Abstract_voter, voter2::Abstract_voter)
    return sum(abs.(get_opinion(voter1) - get_opinion(voter2)))
end