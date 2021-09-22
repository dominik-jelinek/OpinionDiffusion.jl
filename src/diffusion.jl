function evolveEdges!(g, encodedDB, edgeDiffConfig)
    edgeDiffFunc = parseFunction(edgeDiffConfig["edgeDiffFunc"])
    distMetric = parseMetric(edgeDiffConfig["distMetric"])
    
    n = edgeDiffConfig["evolveEdges"]
    start = rand(1:size(encodedDB, 2), n)
    finish = rand(1:size(encodedDB, 2), n)

    for i in n
        v1, v2 = start[i], finish[i]

        distance = Distances.evaluate(distMetric, getCol(encodedDB, v1), getCol(encodedDB, v2))
        if has_edge(g, v1, v2)
            if edgeDiffFunc(distance) < rand()# TO DO maybe open mindedness?
                rem_edge!(g, v1, v2)
            end
        else
            if 1.0 - edgeDiffFunc(distance) <= rand()
                add_edge!(g, v1, v2)
            end
        end
    end
end

function evolvePrefs!(g, encodedDB, vertexDiffConfig)
    canCount = size(encodedDB, 2)
    vertexes = rand(1:canCount, vertexDiffConfig["evolveVertices"])
    for v in vertexes
        neighbors_ = neighbors(g, v)
        if length(neighbors_) == 0
            continue
        end

        neighbor = getCol(encodedDB, neighbors_[rand(1:length(neighbors_))])
        voter = getCol(encodedDB, v)

        if vertexDiffConfig["method"] == "averageOne"
            averageOne!(voter, neighbor)
        elseif vertexDiffConfig["method"] == "averageAll"
            averageAll!(voter, neighbor)
        else
            error("Unknown vertex diffusion method, [averageOne | averageAll]")
        end
    end
end