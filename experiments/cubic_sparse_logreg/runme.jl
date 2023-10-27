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

sigm(z) = 1 / (1 + exp(-z))

struct Cubic{TQ,Tq,R}
    Q::TQ
    q::Tq
    c::R
end

function AdaProx.eval_with_pullback(f::Cubic, x)
    grad = f.Q * x + f.q + (norm(x) * f.c / 2) * x
    cubic_pullback() = grad
    return (dot(x, grad) + dot(f.q, x)) / 2 - norm(x)^3 * f.c / 12, cubic_pullback
end

(f::Cubic)(x) = AdaProx.eval_with_pullback(f, x)[1]

function logistic_loss_grad_Hessian(X, y, w)
    probs = sigm.(X * w[1:end-1] .+ w[end])
    N = size(y, 1)
    g = X' * (probs - y) ./ N
    push!(g, mean(probs - y))  # for bias: X_new = [X, 1] 
    sb = probs .* (1 .- probs) ./ N
    R = diagm(0 => sb)
    XR = X' * R * ones(N, 1)
    # Hessian: H = [X'*R*X XR;XR' sum(sb)]
    H = vcat(hcat(X' * R * X, XR), hcat(XR', sum(sb)))
    return H, g
end

function run_cubic_logreg_data(
    filename,
    ::Type{T} = Float64;
    seed = 0,
    tol = 1e-5,
    maxit = 1000,
    lam = 1.0,
) where {T}
    @info "Start cubic subproblem for L1 Logistic Regression ($filename)"

    Random.seed!(seed)

    X, y = load_libsvm_dataset(filename, T, labels = [0.0, 1.0])

    m, n = size(X)
    n = n + 1

    x0 = zeros(n)

    Q, q = logistic_loss_grad_Hessian(X, y, x0)
    f = Cubic(Q, q, lam)
    g = ProximalCore.Zero()


    x0 = zeros(n)
    x_pert = x0 + randn(size(x0))

    grad_x, _ = ProximalCore.gradient(f, x0)
    grad_x_pert, _ = ProximalCore.gradient(f, x_pert)
    gam_init = norm(x0 - x_pert)^2 / dot(grad_x - grad_x_pert, x0 - x_pert) 

    # run algorithm with 1/10 the tolerance to get "accurate" solution
    sol, numit = AdaProx.adaptive_proxgrad(
        x0,
        f = f,
        g = g,
        rule = AdaProx.OurRule(gamma = gam_init),
        tol = tol / 10,
        maxit = maxit * 10,
        name = nothing,
    )
    xi_values = [1, 1.5, 2]
    for xi = xi_values
        sol, numit = AdaProx.backtracking_proxgrad(
            x0,
            f = AdaProx.Counting(f),
            g = g,
            gamma0 = gam_init,
            xi = xi,
            tol = tol,
            maxit = maxit,
            name = "PGM (backtracking)-(xi=$(xi))",
        )
    end

    sol, numit = AdaProx.backtracking_nesterov(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        gamma0 = gam_init,
        tol = tol,
        maxit = maxit,
        name = "Nesterov (backtracking)",
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.MalitskyMishchenkoRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (MM)",
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.OurRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (Ours)",
    )

    sol, numit = AdaProx.agraal(
        x0,
        f = AdaProx.Counting(f),
        g = g,
        gamma0 = gam_init,
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
        title = "Cubic regularization ($(basename(path)))",
        xlabel = L"# \mbox{ of call to } Q",
        ylabel = L"F(x^k) - F_\star",
    )

    for k in keys(gb)
        if k.method === nothing
            continue
        end
        plot!(
            # each evaluation of f is one mul with Q
            gb[k][!, :f_evals],
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
        run_cubic_logreg_data(
            joinpath(@__DIR__, "..", "datasets", "mushrooms"),
            lam = 1,
            maxit = 100,
            tol = 1e-7,
        )
    end
    plot_convergence(path)

    path = joinpath(@__DIR__, "a5a.jsonl")
    with_logger(get_logger(path)) do
        run_cubic_logreg_data(
            joinpath(@__DIR__, "..", "datasets", "a5a"),
            lam = 1,
            maxit = 100,
            tol = 1e-7,
        )
    end
    plot_convergence(path)

    path = joinpath(@__DIR__, "phishing.jsonl")
    with_logger(get_logger(path)) do
        run_cubic_logreg_data(
            joinpath(@__DIR__, "..", "datasets", "phishing"),
            lam = 1,
            maxit = 100,
            tol = 1e-7,
        )
    end
    plot_convergence(path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
