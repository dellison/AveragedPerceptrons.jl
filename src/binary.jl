"""
    Perceptron

Binary perceptron.
"""
mutable struct Perceptron{T,N} <: AbstractPerceptron{T}
    w::T
    b::N
end

Perceptron(nfeats::Int) = Perceptron(zeros(Int,nfeats), 0)
Perceptron(w::AbstractVector) = Perceptron(w, zero(eltype(w)))
Perceptron(w::Dict{T,N}) where {T,N<:Number} = Perceptron(w,zero(N))

@percep Perceptron

predict(p::Perceptron, x) = score(p, x) > 0

score(p::Perceptron, x) = dot(p.w, x) + p.b
score(p::Perceptron{Dict,T}, x::Dict) where T = sum(get(p.w, k, zero(T)) * x[k]) + p.b
score(p::Perceptron{D,T}, x) where {D<:Dict,T} = sum(get(p.w, k, zero(T)) for k in x) + p.b

function fit_one!(p::Perceptron, x, y::Bool, r=1)
    ŷ = predict(p, x)
    if ŷ != y
        update!(p, x, y, r)
    end
    ŷ
end

function update!(p::Perceptron, x, y::Bool, r=1)
    !y && (r *=-1)
    p.b += r
    p.w .+= (x * r)
    p
end

"""
    SparsePerceptron

Binary perceptron with a sparse coefficient matrix.
"""
const SparsePerceptron{T} = Perceptron{SparseVector,T}

function SparsePerceptron{T}(nfeatures::Int) where T
    w = spzeros(T, nfeatures)
    b = zero(T)
    Perceptron(w, b)
end
SparsePerceptron(nfeatures::Int) =
    SparsePerceptron{Int}(nfeatures)

"""
    DictPerceptron

Averaged perceptron with a Dict representing weights.
"""
const DictPerceptron{F,V} = Perceptron{Dict{F,V},V}

"""
    DictPerceptron()

todo
"""
DictPerceptron() = Perceptron(Dict{Any,Float64}(),zero(Float64))

function update!(p::DictPerceptron{K,V}, x, y::Bool, r=1) where {K,V}
    !y && (r *=-1)
    p.b += r
    for k in x
        p.w[k] = get(p.w, k, zero(V)) + r
    end
    p
end
function update!(p::DictPerceptron{K,V}, x::Dict, y::Bool, r=1) where {K,V}
    !y && (r *=-1)
    p.b += r
    for (k,v) in x
        p.w[k] = get(p.w, k, zero(V)) + v*r
    end
    p
end


"""
    AveragedPerceptron

Binary averaged perceptron.
"""
mutable struct AveragedPerceptron{W,T} <: AbstractPerceptron{T}
    t::Int
    p::Perceptron{W,AveragedWeight{T}}
end

function AveragedPerceptron(nfeats::Int)
    w = [AveragedWeight() for _=1:nfeats]
    b = AveragedWeight()
    AveragedPerceptron(0, Perceptron(w, b))
end

@percep AveragedPerceptron

score(p::AveragedPerceptron, x) = _w(score(p.p, x))
predict(p::AveragedPerceptron, x) = predict(p.p, x)

function fit_one!(p::AveragedPerceptron, x, y, r=1)
    p.t += 1
    fit_one!(p.p, x, y, r)
end

update!(p::AveragedPerceptron, x, y::Bool, r=1) = update!(p.p, x, y, r)

function average!(p::AveragedPerceptron)
    for w in param, param in (p.w, p.b)
        average!(w, p.t)
    end
end

"""
    SparseAveragedPerceptron

Binary averaged perceptron with a sparse coefficient matrix.
"""
const SparseAveragedPerceptron{T} = AveragedPerceptron{SparseVector,T}

function SparseAveragedPerceptron{T}(nfeatures::Int) where T
    w = spzeros(AveragedWeight{T}, nfeatures)
    b = AveragedWeight(T)
    AveragedPerceptron(0, Perceptron(w, b))
end
SparseAveragedPerceptron(nfeatures::Int) =
    SparseAveragedPerceptron{Int}(nfeatures)

"""
   DictAveragedPerceptron

todo
"""
const DictAveragedPerceptron{F,V} = AveragedPerceptron{Dict{F,V},V}

"""
    DictAveragedPerceptron()

todo
"""
function DictAveragedPerceptron()
    w = Dict{Any,AveragedWeight{Float64}}()
    b = AveragedWeight(Float64)
    # {typeof(w),typeof(b)}
    return AveragedPerceptron(0, Perceptron(w, b))
end

"""
    DictAveragedPerceptron{K,V}()

todo
"""
DictAveragedPerceptron{K,V}() where {K,V<:Number} =
    AveragedPerceptron{Dict{K,V},V}(Dict{K,V}(),zero(V))
