@testset "Multiclass Perceptrons" begin

    p = MulticlassPerceptron(10, 3)

    x, y = zeros(10), 1
    @test scores(p, x) == zeros(3)
    ŷ = predict(p, x)
    @test ŷ == y

    x, y = [1,1,1,0,0,0,0,0,0,0], 2
    @test scores(p, x) == zeros(3)
    @test predict(p, x) == 1

    fit_one!(p, x, y)
    @test scores(p, x) == [-4,4,0]
    @test predict(p, x) == 2
end
