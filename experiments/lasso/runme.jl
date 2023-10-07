include(joinpath(@__DIR__, "..", "logging.jl"))

using Random
using LinearAlgebra
using Logging: with_logger, @logmsg
using Tables
using DataFrames
using Plots
using LaTeXStrings
using ProximalCore
using ProximalOperators: NormL1
using AdaProx

pgfplotsx()

struct LinearLeastSquares{TA,Tb}
    A::TA
    b::Tb
end

(f::LinearLeastSquares)(w) = 0.5 * norm(f.A * w - f.b)^2

function ProximalCore.gradient!(grad, f::LinearLeastSquares, w)
    res = f.A * w - f.b
    grad .= f.A' * res
    return 0.5 * norm(res)^2
end

function run_random_lasso(;
    m = 400,
    n = 1000,
    pfactor = 5,
    seed = 0,
    tol = 1e-5,
    kappa = 0.01,
    maxit = 10_000,
)
    @info "Start Lasso ($m by $n)"

    Random.seed!(seed)

    T = Float64
    I = Int64

    p = n / pfactor # nonzeros
    rho = 1 # some positive number controlling how large solution is
    lam = 1  
    y_star = rand(m)
    y_star ./= norm(y_star) #y^\star
    C = rand(m, n) .* 2 .- 1

    CTy = abs.(C' * y_star)
    perm = sortperm(CTy, rev = true) # indices with decreasing order by abs

    alpha = zeros(n)
    for i = 1:n
        if i <= p
            alpha[perm[i]] = lam / CTy[perm[i]]
        else
            temp = CTy[perm[i]]
            if temp < 0.1 * lam
                alpha[perm[i]] = lam
            else
                alpha[perm[i]] = lam * rand() / temp
            end
        end
    end
    A = C * diagm(0 => alpha)   # scaling the columns of Cin
    # generate the primal solution
    x_star = zeros(n)
    for i = 1:n
        if i <= p
            x_star[perm[i]] = rand() * rho / sqrt(p) * sign(dot(A[:, perm[i]], y_star))
        end
    end
    b = A * x_star + y_star
    optimum = norm(y_star) / 2 + lam * norm(x_star, 1) # the solution

    @logmsg AdaProx.Record "" method=nothing it=1 objective=optimum

    Lf = opnorm(A)^2
    gam_init = 1 / Lf
    f = LinearLeastSquares(A, b)
    g = NormL1(lam)

    sol, numit = AdaProx.fixed_proxgrad(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        gamma = gam_init,
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
            gamma0 = gam_init,
            xi = xi, #increase in stepsize
            tol = tol,
            maxit = maxit,
            name = "PGM (backtracking)-(xi=$(xi))"
        )
    end

    sol, numit = AdaProx.backtracking_nesterov(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        gamma0 = gam_init,
        tol = tol,
        maxit = maxit,
        name = "Nesterov (backtracking)"
    )

    sol, numit = AdaProx.backtracking_nesterov_2013(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        gamma = gam_init,
        tol = tol,
        maxit = maxit,
        name = "Nesterov (2013)"
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.MalitskyMishchenkoRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (MM)"
    )

    sol, numit = AdaProx.adaptive_proxgrad(
        zeros(n),
        f = AdaProx.Counting(f),
        g = g,
        rule = AdaProx.OurRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        name = "AdaPGM (Ours)"
    )

    @info "Running aGRAAL"
    sol, numit = AdaProx.agraal(
        zeros(n),
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
        title = "Lasso ($(basename(path)))",
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
    col = [
        (100, 300, 10),
        (500, 1000, 10),
        (4000, 1000, 10),
    ]
    for (m, n, pf) in col
        path = joinpath(@__DIR__, "lasso_$(m)_$(n)_$(pf).jsonl")
        with_logger(get_logger(path)) do
            run_random_lasso(
                m = m,
                n = n,
                pfactor = pf,
                maxit = 2000,
                tol = 1e-7,
                seed = 0,
            )
        end
        plot_convergence(path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
