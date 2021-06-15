using Test

@testset "kendellTau.jl" begin
   @test 3.0 == kendallTau!([1,2,3],[3,2,1],0.5,[0,0,0],normalize = false)
   @test 1.0 == kendallTau!([1,2,3],[1,3,2],0.5,[0,0,0], false)

   @test 0.0 == kendallTau!([1,2],[1], 0.5, [0,0], false)
   @test 1.0 == kendallTau!([2,1],[1], 0.5, [0,0], false)
   @test 0.0 == kendallTau!([1], [1,2], 0.5, [0,0], false)
   @test 1.0 == kendallTau!([1], [2,1], 0.5, [0,0], false)

   @test 2.0 == kendallTau!([1,2],[2,3], 0.5,[0,0,0], false)
   @test 3.0 == kendallTau!([1,3],[2,3], 0.5,[0,0,0], false)
   @test 1.0 == kendallTau!([2,1],[2,3], 0.5,[0,0,0], false)

   @test 4.0 == kendallTau!([1,2],[3,4], 0.0,[0,0,0,0], false)
   @test 5.0 == kendallTau!([1,2],[3,4], 0.5,[0,0,0,0], false)
   @test 6.0 == kendallTau!([1,2],[3,4], 1.0,[0,0,0,0], false)
end

@testset "diffusion.jl/swaps" begin
   a = [5,2,3]
   oneStepRight!(a,1,3)
   @test a == [2,5,3]

   a = [5,2,3]
   oneStepLeft!(a,1,3)
   @test a == [5,3,2]

   a = [5,0,0]
   oneStepRight!(a,1,2)
   @test a == [2,5,0]
   a = [5,0,0]
   oneStepLeft!(a,1,2)
   @test a == [2,5,0]

   a = [5,1,0]
   oneStepRight!(a,1,2)
   @test a == [1,5,0]
   a = [5,1,0]
   oneStepLeft!(a,1,2)
   @test a == [5,2,1]
end

@testset "diffusion.jl/swaps2" begin
   a = [5,2,3]
   oneStepRight!2(a,5,3)
   @test a == [2,5,3]

   a = [5,2,3]
   oneStepLeft!2(a,5,3)
   @test a == [5,3,2]

   a = [5,0,0]
   oneStepRight!2(a,5,2)
   @test a == [2,5,0]
   a = [5,0,0]
   oneStepLeft!2(a,5,2)
   @test a == [2,5,0]

   a = [5,1,0]
   oneStepRight!2(a,5,2)
   @test a == [1,5,0]
   a = [5,1,0]
   oneStepLeft!2(a,5,2)
   @test a == [5,2,1]
end
#avgKT = (0 + 2*1*1.0/6 + 2*1*3.0/6 + 2*1*5.0/6 + 2/6 + 5/6 + 2.5/6)/10
