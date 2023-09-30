include(joinpath(@__DIR__, "..", "libsvm.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))

using Random
using LinearAlgebra
using Statistics
using Logging: with_logger
using Tables
using DataFrames
using Plots
using LaTeXStrings
using ProximalCore
using ProximalOperators: NormL1
using AdaProx

pgfplotsx()

struct LogisticLoss{TX,Ty}
    X::TX
    y::Ty
end

function (f::LogisticLoss)(w)
    z = f.X * w[1:end-1] .+ w[end]
    return -mean((f.y .- 1) .* z .- log.(1 .+ exp.(-z)))
end

function ProximalCore.gradient!(grad, f::LogisticLoss, w)
    z = f.X * w[1:end-1] .+ w[end]
    w = (1 .+ exp.(-z))
    probs = 1 ./ w
    N = size(f.y, 1)
    grad[1:end-1] .= f.X' * (probs - f.y) ./ N
    grad[end] = mean(probs - f.y)
    return -mean((f.y .- 1) .* z .- log.(w))
end

function run_logreg_l1_data(
    filename,
    ::Type{T} = Float64;
    lam,
    tol = 1e-5,
    maxit = 1000,
) where {T}
    @info "Start L1 Logistic Regression ($filename)"

    X, y = load_libsvm_dataset(filename, T, labels = [0.0, 1.0])

    m, n = size(X)
    n = n + 1

    f = LogisticLoss(X, y)
    g = NormL1(T(lam))

    X1 = [X ones(m)]
    Lf = norm(X1 * X1') / 4 / m

    x0 = zeros(n)
    # run algorithm with 1/10 the tolerance to get "accurate" solution
    sol, numit = AdaProx.adaptive_proxgrad(
        x0,
        f = f,
        g = g,
        rule = AdaProx.OurRule(gamma = 1.0),
        tol = tol / 10,
        maxit = maxit * 10,
        name = nothing
    )

    sol, numit = AdaProx.fixed_proxgrad(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        gamma = 1.0 / Lf,
        tol = tol,
        maxit = maxit,
        name = "PGM (1/Lf)"
    )

    xi_values = [1, 1.5, 2]
    for xi = xi_values
        sol, numit = AdaProx.backtracking_proxgrad(
            zeros(n),
            f = AdaProx.Counting(f),
            g = g,
            gamma0 = 5.0,
            xi = xi, #increase in stepsize
            tol = tol,
            maxit = maxit/2,
            name = "PGM (backtracking)-(xi=$(xi))"
        )
    end

    sol, numit = AdaProx.backtracking_nesterov(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        gamma0 = 5.0,
        tol = tol,
        maxit = maxit/2,
        name = "Nesterov (backtracking)"
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.MalitskyMishchenkoRule(gamma = 1.0),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (MM)"
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.OurRule(gamma = 1.0),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (Ours)"
    )

    sol, numit = AdaProx.agraal(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        tol = tol,
        maxit = maxit,
        name = "aGRAAL"
    )
end

function plot_convergence(path)
    df = eachline(path) .|> JSON.parse |> Tables.dictrowtable |> DataFrame
    optimal_value = minimum(df[!, :objective])
    gb = groupby(df, :method)

    fig = plot(
        title = "Logistic regression ($(basename(path)))",
        xlabel = L"\mbox{call to } \mathcal A, \mathcal A'",
        ylabel = L"F(x^k) - F_\star",
    )

    for k in keys(gb)
        if k.method === nothing
            continue
        end
        plot!(
            2*gb[k][!, :grad_f_evals] + gb[k][!, :f_evals],
            max.(1e-14, gb[k][!, :objective] .- optimal_value),
            yaxis = :log,
            label = k.method,
        )
    end

    savefig(fig, joinpath(@__DIR__, "$(basename(path)).pdf"))
end

function main()
    path = joinpath(@__DIR__, "mushrooms.jsonl")
    with_logger(get_logger(path)) do
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "mushrooms"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)

    path = joinpath(@__DIR__, "a5a.jsonl")
    with_logger(get_logger(path)) do
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "a5a"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)

    path = joinpath(@__DIR__, "phishing.jsonl")
    with_logger(get_logger(path)) do
        run_logreg_l1_data(
            joinpath(@__DIR__, "..", "datasets", "phishing"),
            lam = 0.01, maxit = 2000, tol = 1e-7
        )
    end
    plot_convergence(path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
