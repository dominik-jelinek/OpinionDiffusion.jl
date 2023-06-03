using Profile
using BenchmarkTools

using PooledArrays

parties, candidates, database = initDB("ED-00001-00000003.soi")
@time pooled = PooledArray(database)
@time normal = Array(database)

a = [1, 2]
b = repeat(a, 1, 1000000)
