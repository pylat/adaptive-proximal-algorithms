using LinearAlgebra
using ProximalCore

mutable struct Counting{F,I}
    f::F
    count_eval::I
    count_prox::I
    count_gradient::I
    count_mul::I
    count_amul::I
end

ProximalCore.is_convex(::Type{<:Counting{F}}) where {F} = ProximalCore.is_convex(F)
ProximalCore.is_generalized_quadratic(::Type{<:Counting{F}}) where {F} =
    ProximalCore.is_generalized_quadratic(F)

Counting(f::F) where {F} = begin
    count = 0
    Counting{F,typeof(count)}(f, count, count, count, count, count)
end

function (g::Counting)(x)
    g.count_eval += 1
    g.f(x)
end

function ProximalCore.gradient!(grad, g::Counting, x)
    g.count_gradient += 1
    ProximalCore.gradient!(grad, g.f, x)
end

function ProximalCore.prox!(y, g::Counting, x, gamma)
    g.count_prox += 1
    ProximalCore.prox!(y, g.f, x, gamma)
end

struct AdjointOperator{O}
    op::O
end

LinearAlgebra.norm(C::Counting) = norm(C.f)
LinearAlgebra.adjoint(C::Counting) = AdjointOperator(C)

function Base.:*(C::Counting, x)
    C.count_mul += 1
    return C.f * x
end

function Base.:*(A::AdjointOperator{<:Counting}, x)
    A.op.count_amul += 1
    return A.op.f' * x
end

record_grad_count(_) = nothing
record_grad_count(c::Counting) = c.count_gradient

record_prox_count(_) = nothing
record_prox_count(c::Counting) = c.count_prox

record_mul_count(_) = nothing
record_mul_count(c::Counting) = c.count_mul

record_amul_count(_) = nothing
record_amul_count(c::Counting) = c.count_amul

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