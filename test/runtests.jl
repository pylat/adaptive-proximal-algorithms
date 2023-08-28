using Test
using AdaProx
using ProximalCore

struct Simple2DObjective end

(::Simple2DObjective)(x) = log(1 + x[1]^2)^2 + 10 * x[2]^2

function ProximalCore.gradient!(grad, f::Simple2DObjective, x)
    grad .= [2 * log(1 + x[1]^2) * 2 * x[1] / (1 + x[1]^2), 20 * x[2]]
    return f(x)
end

struct Simple2DBox end

(::Simple2DBox)(x) = abs(x[1]) <= 2.9 ? zero(eltype(x)) : eltype(x)(Inf)

function ProximalCore.prox!(y, ::Simple2DBox, x, _)
    y[1] = clamp(x[1], -2.9, +2.9)
    y[2] = x[2]
    return zero(eltype(x))
end

@testset "Simple 2D problem" begin
    f = Simple2DObjective()
    g = Simple2DBox()

    obj_tol = 1e-7

    sol, numit = AdaProx.adaptive_proxgrad(
        ones(2), f=f, g=g, rule=AdaProx.OurRule(gamma=1.0),
    )

    @test f(sol) < obj_tol
    @test iszero(g(sol))

    sol, numit = AdaProx.backtracking_proxgrad(
        ones(2), f=f, g=g, gamma0=1.0,
    )

    @test f(sol) < obj_tol
    @test iszero(g(sol))

    sol, numit = AdaProx.backtracking_nesterov(
        ones(2), f=f, g=g, gamma0=1.0,
    )

    @test f(sol) < obj_tol
    @test iszero(g(sol))
end
