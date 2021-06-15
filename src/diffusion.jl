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

#=
function evolvePrefs!(g, database, encodedDB, distances, vertexDiffConfig)
    vertexes = rand(1:size(database, 2), vertexDiffConfig["evolveVertices"])
    for v in vertexes
        neighbors_ = neighbors(g, v)
        if length(neighbors_) == 0
            continue
        end

        neighbor = neighbors_[rand(1:length(neighbors_))]

        if rand() < 0.5 # TO DO: add stubbornness
            # changing neighbor
            changeNeighbor!(database, encodedDB, v, neighbor)
            getCol(encodedDB, neighbor) .= encode(getCol(database, neighbor))
            updateDistances(encodedDB, distances, neighbor, metric)
        else
            # changing v
            changeNeighbor!(database, encodedDB, neighbor, v)
            getCol(encodedDB, v) .= encode(getCol(database, v))
            updateDistances(encodedDB, distances, v, metric)
        end
    end
end



"""
Gets all swaps in between 2 preferences and chooses one.
The left(right) candidate in reversed pair is moved to the right(left) by one position
"""
function changeNeighbor!(database, encodedDB, v, neighbor)
    neighborPref = getCol(database, neighbor)
    
    # pick a swap
    swaps = getAllSwaps(getCol(encodedDB, v), neighborPref)
    if length(swaps) == 0
        return
    end
    (can1, can2), val = swaps[rand(1:length(swaps))]

    idxCan1 = getIndex(neighborPref, can1)
    idxCan2 = getIndex(neighborPref, can2)
    
    # check if candidates are in wrong order
    if idxCan1 > idxCan2
        can1, can2 = can2, can1
        idxCan1, idxCan2 = idxCan2, idxCan1
    end

    if abs(val) == 2
        # neighbor moves his preference towards v by one step
        if rand() < 0.5
            swapRight!(neighborPref, idxCan1)
        else
            swapLeft!(neighborPref, idxCan2)
        end
    elseif idxCan1 == idxCan2
        # neighbor makes idea about more favorable candidate in v
        if val == 1
            swapLeft!(neighborPref, idxCan1)
        else
            swapLeft!(neighborPref, idxCan2)
        end
    else # abs(val) == 1
        # v hasn't got a preference for can1 and can2
        return
    end
end

function changeNeighbor2!(database, encodedDB, v, neighbor, weights)
    # pick random candidate
    can = rand(1:size(database, 1))

    prefV = getCol(database, v)
    prefNei = getCol(database, neighbor)

    idxCanV = getIndex(prefV, can)
    idxCanNei = getIndex(prefNei, can)
    if prefV[idxCanV] == 0
        return # no swap, unknown preference in v
    end

    encodedNei = getCol(encodedDB, neighbor)
    # check if can is to the left or right in neighbor relative to v
    if idxCanV < idxCanNei
        # moving neighbor left
        canWeight = encodedNei[can]
        leftWeight = encodedNei[prefNei[idxCanNei - 1]]
        if rand() < 1 - (leftWeight + canWeight) / 2
            swapLeft!(prefNei, can, idxCanNei)
            encodedNei .= spearmanEncoding(prefNei, weights)
        end
    elseif idxCanV > idxCanNei
        # moving neighbor right
        canWeight = encodedNei[can]
        rightWeight = encodedNei[prefNei[idxCanNei + 1]]
        if rand() < 1 - (canWeight + rightWeight) / 2
            swapRight!(prefNei, idxCanNei)
            encodedNei .= spearmanEncoding(prefNei, weights)
        end
    else
        # no swap, unknown or the same position
    end
end

function changeNeighbor3!(database, encodedDB, v, neighbor, weights)
    # pick random candidates
    can1, can2 = StatsBase.sample(1:size(database, 1), 2)

    if encodedDB[can1, v] == encodedDB[can2, v]
        return # no swap, unknown preference in v
    end

    if encodedDB[can1, v] < encodedDB[can2, v]
        can1, can2 = can2, can1
    end
    
    prefV = getCol(database, v)
    prefNei = getCol(database, neighbor)
    idxCanV = getIndex(prefV, can)
    idxCanNei = getIndex(prefNei, can)

    encodedNei = getCol(encodedDB, neighbor)
    # check if can is to the left or right in neighbor relative to v
    if rand() < 0.5
        swapRight!(neighborPref, idxCan1)
    else
        swapLeft!(neighborPref, idxCan2)
    end
end

"""
Move candidate that is first out of can1 and can2 in the preference one position to the right
if can1 = 1 can2 = 3 (1,2,3) -> (2,1,3)
if can1 = 1 can2 = 2 (1,0,0) -> (2,1,0)
"""
function swapRight!(toChange, can2, idxCan1)
    # can1 or can2 must be in toChange

    # move to right
    if toChange[idxCan1 + 1] == 0
        toChange[idxCan1], toChange[idxCan1 + 1] = can2, toChange[idxCan1]
    else
        toChange[idxCan1], toChange[idxCan1 + 1] = toChange[idxCan1 + 1], toChange[idxCan1]
    end
end

"""
Move candidate that is further in the preference one position to the left
if can1 = 1 can2 = 3 (1,2,3) -> (2,3,1)
if can1 = 1 can2 = 3 (1,2,0) -> (1,2,3)
"""
function swapLeft!(toChange, can, idxCan)
    # can1 or can2 must be in toChange and idxCan1 < idxCan2

    # move to left
    if toChange[idxCan] == 0
        toChange[idxCan] = can
    else
        toChange[idxCan - 1], toChange[idxCan] = toChange[idxCan], toChange[idxCan - 1]
    end
end
=#