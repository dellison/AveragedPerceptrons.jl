using AveragedPerceptrons, Test

@testset "AveragedPerceptrons.jl" begin
    using AveragedPerceptrons: fit_one!, update!, predict, score, scores
    include("test_binary_perceptrons.jl")
    include("test_multiclass_perceptrons.jl")

    using AveragedPerceptrons: averaged, average!
    include("test_averaged_perceptrons.jl")
end
