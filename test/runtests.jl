using Test
using LinearAlgebra
using AdaProx
using ProximalCore

struct Simple2DObjective end

function AdaProx.eval_with_pullback(::Simple2DObjective, x)
    simple_2d_pullback() = [2 * log(1 + x[1]^2) * 2 * x[1] / (1 + x[1]^2), 20 * x[2]]
    return log(1 + x[1]^2)^2 + 10 * x[2]^2, simple_2d_pullback
end

(f::Simple2DObjective)(x) = AdaProx.eval_with_pullback(f, x)[1]

struct Simple2DBox end

(::Simple2DBox)(x) = abs(x[1]) <= 2.9 ? zero(eltype(x)) : eltype(x)(Inf)

function ProximalCore.prox!(y, ::Simple2DBox, x, _)
    y[1] = clamp(x[1], -2.9, +2.9)
    y[2] = x[2]
    return zero(real(eltype(x)))
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
        ones(2), f=f, g=g, gamma0=1.0, xi=1.1,
    )

    @test f(sol) < obj_tol
    @test iszero(g(sol))

    sol, numit = AdaProx.backtracking_nesterov(
        ones(2), f=f, g=g, gamma0=1.0,
    )

    @test f(sol) < obj_tol
    @test iszero(g(sol))
end

@testset "Counting" begin
    f = AdaProx.Counting(Simple2DObjective())
    g = AdaProx.Counting(Simple2DBox())
    A = AdaProx.Counting(Matrix(I, 2, 2))

    x = ones(2)

    _, pb = AdaProx.eval_with_pullback(f, x)
    _, _ = ProximalCore.prox(g, x)
    _, = A * x

    @test f.eval_count == 1
    @test f.grad_count == 0
    @test g.prox_count == 1
    @test A.mul_count == 1
    @test A.amul_count == 0

    pb()

    @test f.grad_count == 1

    _, = A' * x

    @test A.amul_count == 1

    AdaProx.without_counting() do
        _, pb = AdaProx.eval_with_pullback(f, x)
        pb()
        _, _ = ProximalCore.prox(g, x)
        _, = A * x
    end

    @test f.eval_count == 1
    @test f.grad_count == 1
    @test g.prox_count == 1
    @test A.mul_count == 1
    @test A.amul_count == 1
end
