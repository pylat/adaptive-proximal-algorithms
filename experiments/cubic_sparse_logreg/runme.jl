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

(f::Cubic)(x) = dot(x, f.Q * x) / 2 + dot(x, f.q) + norm(x)^3 * f.c / 6

function ProximalCore.gradient!(grad, f::Cubic, x)
    grad .= f.Q * x + f.q + (f.c * norm(x) / 2) * x
    return (dot(f.q, grad) + dot(f.q, x)) / 2 + norm(x)^3 * f.c / 6
end

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

    x0 = zeros(n, 1)

    Q, q = logistic_loss_grad_Hessian(X, y, x0)
    f = Cubic(Q, q, lam)
    g = ProximalCore.Zero()

    @info "Getting accurate solution"

    sol, numit = AdaProx.adaptive_proxgrad(
        zeros(n),
        f = f,
        g = g,
        rule = AdaProx.OurRule(gamma = 1.0),
        tol = tol / 10,
        maxit = maxit * 10,
        name = nothing,
    )

    @info "Running solvers"

    sol, numit = AdaProx.backtracking_proxgrad(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        gamma0 = 1.0,
        tol = tol,
        maxit = maxit,
        name = "PGM (backtracking)",
    )
    @info "PGM, backtracking step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit = AdaProx.backtracking_nesterov(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        gamma0 = 1.0,
        tol = tol,
        maxit = maxit,
        name = "Nesterov (backtracking)",
    )
    @info "Nesterov PGM, backtracking step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit = AdaProx.adaptive_proxgrad(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.MalitskyMishchenkoRule(gamma = 0.001),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (MM)",
    )
    @info "PGM, MM adaptive step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit = AdaProx.adaptive_proxgrad(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.OurRule(gamma = 1.0),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (Ours)",
    )
    @info "PGM, our adaptive step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit = AdaProx.auto_adaptive_proxgrad(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        gamma= 1e5,
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (Ours)",
    )
    @info "PGM, our auto adaptive step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit = AdaProx.agraal(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        tol = tol,
        maxit = maxit,
        name = "aGRAAL"
    )
    @info "aGRAAL"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"
end

function plot_convergence(path)
    df = eachline(path) .|> JSON.parse |> Tables.dictrowtable |> DataFrame
    optimal_value = minimum(df[!, :objective])
    gb = groupby(df, :method)

    fig = plot(
        title = "Cubic regularization ($(basename(path)))",
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
