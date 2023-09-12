include(joinpath(@__DIR__, "..", "logging.jl"))

using LinearAlgebra
using Logging: with_logger, @logmsg
using Tables
using DataFrames
using ProximalCore
using AdaProx
using Plots
using LaTeXStrings

pgfplotsx()

struct WorstQuadratic{S,T}
    k::S
    L::T
end

function (f::WorstQuadratic)(x)
    s = x[1]^2 + x[f.k]^2
    for i in 1:(f.k-1)
        s += (x[i] - x[i+1])^2
    end
    return (f.L / 4) * (s / 2 - x[1])
end

function ProximalCore.gradient!(grad, f::WorstQuadratic, x)
    grad[1] = (f.L / 4) * (2 * x[1] - x[2] - 1)
    for i in 2:(f.k-1)
        grad[i] = (f.L / 4) * (2 * x[i] - x[i-1] - x[i+1])
    end
    grad[f.k] = (f.L / 4) * (2 * x[f.k] - x[f.k-1])
    grad[(f.k+1):end] .= 0
    # since f is quadratic, f(x) = 1/2 <x, Q x> + <q, x>
    # meaning \nabla f(x) = Q x + q
    # and f(x) = (1/2) <\nabla f(x), x> + (1/2) <q, x>
    # since q = -(L/4) e_1, we obtain the following
    return dot(grad, x) / 2 - (f.L / 8) * x[1]
end

function run_nesterov_worst_case()
    k = 100
    n = 100

    @assert n >= k

    L = 100.0

    f = WorstQuadratic(k, L)
    g = ProximalCore.Zero()

    optimum_value = (L / 8) * (1 / (k + 1) - 1)

    tol = 1e-6
    maxit = 10_000

    sol, numit = AdaProx.fixed_proxgrad(
        zeros(n),
        f=AdaProx.Counting(f),
        g=g,
        gamma=1 / L,
        tol=tol,
        maxit=maxit,
    )

    sol, numit = AdaProx.backtracking_proxgrad(
        zeros(n),
        f=AdaProx.Counting(f),
        g=g,
        gamma0=1.0,
        tol=tol,
        maxit=maxit,
    )

    sol, numit = AdaProx.fixed_nesterov(
        zeros(n),
        f=AdaProx.Counting(f),
        g=g,
        gamma=1 / L,
        tol=tol,
        maxit=maxit,
    )

    sol, numit = AdaProx.backtracking_nesterov(
        zeros(n),
        f=AdaProx.Counting(f),
        g=g,
        gamma0=1.0,
        tol=tol,
        maxit=maxit,
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        zeros(n),
        f=AdaProx.Counting(f),
        g=g,
        rule=AdaProx.MalitskyMishchenkoRule(gamma=1 / L),
        tol=tol,
        maxit=maxit,
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        zeros(n),
        f=AdaProx.Counting(f),
        g=g,
        rule=AdaProx.OurRule(gamma=1 / L),
        tol=tol,
        maxit=maxit,
    )
end

function plot_convergence(path)
    df = eachline(path) .|> JSON.parse |> Tables.dictrowtable |> DataFrame
    optimal_value = minimum(df[!, :objective])
    gb = groupby(df, :method)

    fig = plot(
        title = "Nesterov's worst case",
        xlabel = L"\nabla f\ \mbox{evaluations}",
        ylabel = L"F(x^k) - F_\star",
    )

    for k in keys(gb)
        if k.method === nothing
            continue
        end
        plot!(
            gb[k][!, :grad_f_evals],
            max.(1e-14, gb[k][!, :objective] .- optimal_value),
            yaxis = :log,
            label = k.method,
        )
    end

    savefig(fig, joinpath(@__DIR__, "$(basename(path)).pdf"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    path = joinpath(@__DIR__, "nesterov_worst_case.jsonl")
    with_logger(get_logger(path)) do
        run_nesterov_worst_case()
    end
    plot_convergence(path)
end
