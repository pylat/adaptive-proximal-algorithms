using LinearAlgebra
using ProximalCore

_counting_enabled = true

is_counting_enabled() = _counting_enabled

function without_counting(f)
    global _counting_enabled
    _counting_enabled = false
    ret = f()
    _counting_enabled = true
    return ret
end

mutable struct Counting{F,I}
    f::F
    eval_count::I
    grad_count::I
    prox_count::I
    mul_count::I
    amul_count::I
end

ProximalCore.is_convex(::Type{<:Counting{F}}) where {F} = ProximalCore.is_convex(F)
ProximalCore.is_generalized_quadratic(::Type{<:Counting{F}}) where {F} =
    ProximalCore.is_generalized_quadratic(F)

Counting(f::F) where {F} = begin
    count = 0
    Counting{F,typeof(count)}(f, count, count, count, count, count)
end

(f::Counting)(args...) = f.f(args...)

function eval_with_pullback(g::Counting, x)
    if is_counting_enabled()
        g.eval_count += 1
    end

    f_x, pb = eval_with_pullback(g.f, x)

    function counting_pullback()
        if is_counting_enabled()
            g.grad_count += 1
        end
        return pb()
    end

    return f_x, counting_pullback
end

function ProximalCore.prox!(y, g::Counting, x, gamma)
    if is_counting_enabled()
        g.prox_count += 1
    end

    return ProximalCore.prox!(y, g.f, x, gamma)
end

struct AdjointCounting{O}
    op::O
end

LinearAlgebra.norm(C::Counting) = norm(C.f)
LinearAlgebra.adjoint(C::Counting) = AdjointCounting(C)

function Base.:*(C::Counting, x)
    if is_counting_enabled()
        C.mul_count += 1
    end

    return C.f * x
end

function Base.:*(A::AdjointCounting{<:Counting}, x)
    if is_counting_enabled()
        A.op.amul_count += 1
    end

    return A.op.f' * x
end

grad_count(_) = nothing
grad_count(c::Counting) = c.grad_count

prox_count(_) = nothing
prox_count(c::Counting) = c.prox_count

mul_count(_) = nothing
mul_count(c::Counting) = c.mul_count

amul_count(_) = nothing
amul_count(c::Counting) = c.amul_count

eval_count(_) = nothing
eval_count(c::Counting) = c.eval_count
