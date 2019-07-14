module AveragedPerceptrons

export Perceptron, DictPerceptron, SparsePerceptron
export AveragedPerceptron, DictAveragedPerceptron, SparseAveragedPerceptron
export MulticlassPerceptron, DictAveragedPerceptron, SparseMulticlassPerceptron
export MulticlassAveragedPerceptron, DictMulticlassAveragedPerceptron,
    SparseMulticlassAveragedPerceptron

using LinearAlgebra, SparseArrays

abstract type AbstractPerceptron{T} end

"""
    fit!(p, data, r=1)

Fit a perceptron to a dataset (one iteration).
"""
function fit!(p::AbstractPerceptron, data, r=1)
    for (x, y) in data
        fit_one!(p, x, y, r)
    end
end

macro percep(P)
    @eval (p::$P)(x)    = predict(p, x)
    @eval (p::$P)(x, y) = fit_one!(p, x, y)
end

include("weights.jl")
include("binary.jl")
include("multiclass.jl")

end # module
