@testset "Kernels" begin
    import LikelihoodfreeInference: bandwidth, update!, Bandwidth, RosenblattSchedule, Scheduled
    k = Kernel(type = Gaussian,
               bandwidth = Bandwidth(heuristic = nothing, gamma = 2.))
    @test bandwidth(k) == 2.
    update!(k, 0)
    @test bandwidth(k) == 2.
    @test k(1, 4) == exp(-(4 - 1)^2/(2 * 2^2))
    @test k([2, 0, 1], zeros(3)) ≈ exp(-5/(2*2^2))
    k = Kernel(type = Gaussian,
               bandwidth = Bandwidth(heuristic = nothing, gamma = [2., 6.]))
    @test bandwidth(k) == [2., 6.]
    @test k([3, 0], [4, 2]) == exp(-sum(abs2, ([3, 0] - [4, 2]) ./ [2., 6])/2)
    k = Kernel(type = ModifiedGaussian,
               bandwidth = Bandwidth(heuristic = nothing, gamma = .5))
    @test k(1, 0) == exp(-1/(2*.5))
    @test k(.3, 0) == exp(-.3^2/(2*.5^2))
    k = Kernel(bandwidth = Bandwidth(heuristic = MedianHeuristic()))
    update!(k, [2, 2])
    @test bandwidth(k) == 2
    k = Kernel(bandwidth = Bandwidth(heuristic = MedianHeuristic(.1)))
    update!(k, [2, 2])
    @test bandwidth(k) == .1 * 2
    k = Kernel(bandwidth = Bandwidth(heuristic = ScottsHeuristic()))
    update!(k, [2, 2])
    @test bandwidth(k) == 0
    k = Kernel(bandwidth = Scheduled(Bandwidth(heuristic = nothing, gamma = 2),
                                     RosenblattSchedule(2)))
    update!(k, nothing)
    update!(k, nothing)
    @test bandwidth(k) == 2 * 2^(-1/(2 + 4))
end
