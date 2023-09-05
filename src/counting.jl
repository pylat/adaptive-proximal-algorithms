using LinearAlgebra
using ProximalCore

mutable struct Counting{F,I}
    f::F
    eval_count::I
    prox_count::I
    grad_count::I
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

function (g::Counting)(x)
    g.eval_count += 1
    g.f(x)
end

function ProximalCore.gradient!(grad, g::Counting, x)
    g.grad_count += 1
    ProximalCore.gradient!(grad, g.f, x)
end

function ProximalCore.prox!(y, g::Counting, x, gamma)
    g.prox_count += 1
    ProximalCore.prox!(y, g.f, x, gamma)
end

struct AdjointOperator{O}
    op::O
end

LinearAlgebra.norm(C::Counting) = norm(C.f)
LinearAlgebra.adjoint(C::Counting) = AdjointOperator(C)

function Base.:*(C::Counting, x)
    C.mul_count += 1
    return C.f * x
end

function Base.:*(A::AdjointOperator{<:Counting}, x)
    A.op.amul_count += 1
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

nocount(obj) = obj
nocount(c::Counting) = c.f

function obj(f, g, h, A, x) 
    y = try 
        nocount(f)(x) +
        nocount(g)(x) + 
        nocount(h)(nocount(A) * x)
    catch e 
        nocount(f)(x)
    end 
    return y
end