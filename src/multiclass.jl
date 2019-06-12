"""
    MulticlassPerceptron

Multi-class perceptron classifier.
"""
mutable struct MulticlassPerceptron{T,N} <: AbstractPerceptron{T}
    w::T
    b::Vector{N}
end

MulticlassPerceptron{T}(nfeats::Int, nclasses::Int) where T =
    MulticlassPerceptron(zeros(T, nclasses, nfeats), zeros(T, zeros(nclasses)))

MulticlassPerceptron(nfeats::Int, nclasses::Int) =
    MulticlassPerceptron(zeros(nclasses, nfeats), zeros(nclasses))

@percep MulticlassPerceptron

predict(p, x)                      = p.w * x + p.b |> argmax
scores(p::MulticlassPerceptron, x) = p.w * x + p.b

function fit_one!(p::MulticlassPerceptron, x, y, r=1)
    ŷ = predict(p, x)
    if ŷ != y
        update!(p, x, ŷ, -r)
        update!(p, x, y,  r)
    end
    ŷ
end

function update!(p::MulticlassPerceptron, x, y, r=1)
    p.b[y] += r
    p.w[y, :] .+= x * r
    p
end

function update!(p::MulticlassPerceptron, x::SparseVector, y, r=1)
    p.b[y] += r
    for (i, w) in zip(findnz(x)...)
        p.w[y, i] += r
    end
    p
end

"""
    SparseMulticlassPerceptron

Multi-class perceptron classifier with sparse feature weights.
"""
const SparseMulticlassPerceptron{T} =
    MulticlassPerceptron{SparseMatrixCSC{AveragedWeight{T},Int}}

function SparseMulticlassPerceptron{T}(nfeatures::Int,nclasses::Int) where T
    w = spzeros(T,nclasses,nfeatures)
    b = [zero(T) for _=1:nclasses]
    MulticlassPerceptron(w,b)
end
SparseMulticlassPerceptron(nfeatures::Int,nclasses::Int) =
    SparseMulticlassAveragedPerceptron{Int}(nclasses, nfeatures)

"""
    MulticlassAveragedPerceptron

Multiclass averaged perceptron classifier.
"""
mutable struct MulticlassAveragedPerceptron{W,T} <: AbstractPerceptron{T}
    t::Int
    p::MulticlassPerceptron{W,AveragedWeight{T}}
end

MulticlassAveragedPerceptron(int::Int, out::Int) =
    MulticlassAveragedPerceptron{Int}(int::Int, out::Int)

function MulticlassAveragedPerceptron{T}(in::Int, out::Int) where T
    w = fill(AveragedWeight{T}(), out, in)
    b = fill(AveragedWeight{T}(), out)
    p = MulticlassPerceptron(w, b)
    MulticlassAveragedPerceptron(0, p)
end

@percep MulticlassAveragedPerceptron

"""
    averaged(p)

Average the perceptron's weights, returning a `MulticlassPerceptron`.
"""
function averaged(p::MulticlassAveragedPerceptron)
    t, w, b = p.t, p.p.w, p.p.b
    average(w) = averaged(w, t)
    MulticlassPerceptron(average.(w), average.(b))
end

"""
    average!(p)

Average the perceptron's weights in-place.
"""
function average!(p::MulticlassAveragedPerceptron)
    avg! = w -> average!(w, p.t)
    avg!.(p.p.w)
    avg!.(p.p.b)
    p
end

predict(p::MulticlassAveragedPerceptron, x) = predict(p.p, x)
scores(p::MulticlassAveragedPerceptron,  x) = scores(p.p, x)
update!(p::MulticlassAveragedPerceptron, x, y, r=1) = update!(p.p, x, y, r)

function fit_one!(p::MulticlassAveragedPerceptron, x, y, r=1)
    p.t += 1
    fit_one!(p.p, x, y, r)
end

"""
    SparseMulticlassAveragedPerceptron

Multi-class averaged perceptron classifier with a sparse coefficient matrix.
"""
const SparseMulticlassAveragedPerceptron{T} = MulticlassAveragedPerceptron{SparseMatrixCSC{AveragedWeight{T},Int}}

function SparseMulticlassAveragedPerceptron{T}(nfeatures::Int,nclasses::Int) where T
    w = spzeros(AveragedWeight{T},nclasses,nfeatures)
    b = [AveragedWeight(T) for _=1:nclasses]
    MulticlassAveragedPerceptron(0,MulticlassPerceptron(w,b))
end
SparseMulticlassAveragedPerceptron(nfeatures::Int,nclasses::Int) =
    SparseMulticlassAveragedPerceptron{Int}(nfeatures, nclasses)

