@testset "Binary Averaged Perceptrons" begin

    p = AveragedPerceptron(5)

    x, y = [0,0,0,0,0], true
    @test p.p.b == 0
    @test p.p.w == zeros(5)
    @test score(p, x) == 0
    @test predict(p, x) == false
    fit_one!(p, x, y)

    x, y = [1,0,0,0,0], false
    @test p.p.b == 1
    @test p.p.w == [0,0,0,0,0]
    @test score(p, x) == 1
    @test predict(p, x) == true
    fit_one!(p, x, y)
    @test p.p.b == 0
    @test p.p.w == [-1,0,0,0,0]
    @test score(p, x) == -1

    x, y = [1,2,0,1,0], false
    @test score(p, x) == -1
    @test predict(p, x) == false
end

@testset "Multiclass Averaged Perceptrons" begin

    p = MulticlassAveragedPerceptron(5, 3)

    p2 = MulticlassAveragedPerceptron(5, 3)
    @test p2.p.w == p.p.w && p2.p.b == p.p.b

    x, y = [1,0,0,0,0], 2
    @test p.p.b == [0,0,0]
    @test p.p.w == zeros(3, 5)
    @test scores(p, x) == [0,0,0]
    @test predict(p, x) == 1
    fit_one!(p, x, y)
    @test p.p.b == [-1,1,0]
    @test p.p.w == [-1 1 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0]'
    @test scores(p, x) == p.p.w * x + p.p.b == [-4,4,0]
    @test predict(p, x) == y == 2

    x, y = [1,2,3,0,0], 1
    @test scores(p, x) == p.p.w * x + p.p.b
    @test predict(p, x) == 2
    fit_one!(p, x, y)
    @test predict(p, x) == 1

    @test scores(p, x) == [13, -13, 0]
    p2 = averaged(p)
    average!(p)
    @test round.(p.p.w) == round.(p2.w)
    @test round.(p.p.b) == round.(p2.b)
end

@testset "Sparse Multiclass Averaged Perceptrons" begin
    using SparseArrays
    V,T = 10_000, 3
    randx() = sparsevec([rand(1:V) for _=1:rand(45:60)], 1, V)
    randy() = rand(1:T)
    p = SparseMulticlassAveragedPerceptron(V, T)
    for i in 1:5
        x, y = randx(), randy()
        fit_one!(p, x, y)
    end
    average!(p)
end
