import Base.==, Base.+, Base.*, Base.-, Base./
import Base.conj
import Base.isless, Base.round, Base.zero
import Base.convert
import Base.iterate, Base.show

"""
    AveragedWeight

Weight in an averaged perceptron.

It keeps track of its sum over time as it's updated, and can be
efficiently averaged at the end of training.
"""
mutable struct AveragedWeight{T <: Number} <: Number
    weight::T
    t::Int
    summed::T
end

const Wt{T} = AveragedWeight{T}

AveragedWeight{T}(w=zero(T),t=0) where T = AveragedWeight{T}(w, t, zero(w))
AveragedWeight(T::Type{<:Number}; w=0, t=0) = AveragedWeight{T}(w,0,zero(T))
AveragedWeight(w=0, t=0) = AveragedWeight(w, t, zero(w))

AveragedWeight(w::AveragedWeight) = w

_w(x) = x
_w(w::AveragedWeight) = w.weight

"""
    tick!(w, t)

Make weight current as of timestamp `t`.
"""
function tick!(w::AveragedWeight, t)
    w.summed += (t - w.t) * w.weight
    w.t = t
end

"""
    update!(w::AveragedWeight, t, r=1)

Increment weight `w` at time `t` by rate `r`.
"""
function update!(w::AveragedWeight, t, r=1)
    tick!(w, t)
    w.weight += r
end

"""
    averaged(w::AveragedWeight, t, r=1) 

Average of weight `w` over its lifetime, at time `t`.
"""
averaged(w::AveragedWeight, t) = (w.summed + (t - w.t) * w.weight) / t

"""
    average!(w::AveragedWeight, t)

Average weight `w` in-place over its lifetime at time `t`.
"""
function average!(w::AveragedWeight, t)
    tick!(w, t)
    avg = w.summed / t
    try
        w.weight = avg
    catch err
        if err isa InexactError
            w.weight = round(typeof(w.weight), avg)
        else
            rethrow(err)
        end
    end
    return w
end

Base.isless(w::AveragedWeight, x) = isless(w.weight, x)
Base.isless(x, w::AveragedWeight) = isless(x, w.weight)
Base.isless(w1::AveragedWeight, w2::AveragedWeight) = isless(w1.weight, w2.weight)

==(w::AveragedWeight, x::Number) = w.weight == x
==(x::Number, w::AveragedWeight) = x == w.weight
==(w::AveragedWeight, w2::AveragedWeight) =
    w.weight == w2.weight && w.t == w2.t && w.summed == w2.summed

# for op in [+, -, *, /]
#     @eval quote
#         $op(w::AveragedWeight, x) = AveragedWeight(_w(w) + _w(x), w.t, w.summed)
#         $op(x, w::AveragedWeight) = AveragedWeight(_w(w) + _w(x), w.t, w.summed)
#         $op(w::AveragedWeight,w2::AveragedWeight) =
#             AveragedWeight($op(_w(w), + _w(w2), w.t, w.summed)

# end
+(w::AveragedWeight, x::Number) = AveragedWeight(w.weight + x, w.t, w.summed)
+(x::Number, w::AveragedWeight) = AveragedWeight(x + w.weight, w.t, w.summed)
+(w::AveragedWeight,w2::AveragedWeight) = AveragedWeight(w.weight + w2.weight, w.t, w.summed)
-(w::AveragedWeight, x::Number) = AveragedWeight(w.weight - x, w.t, w.summed)
-(x::Number, w::AveragedWeight) = AveragedWeight(x - w.weight, w.t, w.summed)
-(w::AveragedWeight,w2::AveragedWeight) = AveragedWeight(w.weight - w2.weight, w.t, w.summed)
*(w::AveragedWeight, x::Number) = AveragedWeight(w.weight * x, w.t, w.summed)
*(x::Number, w::AveragedWeight) = AveragedWeight(x * w.weight, w.t, w.summed)
*(w::AveragedWeight,w2::AveragedWeight) = AveragedWeight(w.weight * w2.weight, w.t, w.summed)
/(w::AveragedWeight, x::Number) = AveragedWeight(w.weight / x, w.t, w.summed)
/(x::Number, w::AveragedWeight) = AveragedWeight(x / w.weight, w.t, w.summed)
/(w::AveragedWeight,w2::AveragedWeight) = AveragedWeight(w.weight / w2.weight, w.t, w.summed)

conj(w::AveragedWeight) = w.weight

# -(w::AveragedWeight, x) = AveragedWeight(_w(w) - _w(x), w.t, w.summed)
# -(x, w::AveragedWeight) = AveragedWeight(_w(w) - _w(x), w.t, w.summed)
# -(w::AveragedWeight,w2::AveragedWeight) =
#     AveragedWeight(_w(w) - _w(w2), w.t, w.summed)

# *(w::AveragedWeight, x) = AveragedWeight(_w(w) * _w(x), w.t, w.summed)
# *(x, w::AveragedWeight) = AveragedWeight(_w(x) * _w(w), w.t, w.summed)
# *(w::AveragedWeight,w2::AveragedWeight) =
#     AveragedWeight(_w(w) * _w(w2), w.t, w.summed)

# /(w::AveragedWeight, x) = AveragedWeight(_w(w) / _w(x), w.t, w.summed)
# /(x, w::AveragedWeight) = AveragedWeight(_w(w) / _w(x), w.t, w.summed)
# /(w::AveragedWeight,w2::AveragedWeight) =
#     AveragedWeight(_w(w) / _w(w2), w.t, w.summed)

# Base.convert(T, w::AveragedWeight) = T(w.weight)

Base.round(w::AveragedWeight) = AveragedWeight(round(w.weight), w.t, w.summed)
Base.zero(T::AveragedWeight) = AveragedWeight(T)
Base.zero(T::Type{<:AveragedWeight}) = T()

Base.iterate(w::AveragedWeight, state...) = iterate(w.weight, state...)

Base.show(io::IO, w::AveragedWeight) = print(io, w.weight)
