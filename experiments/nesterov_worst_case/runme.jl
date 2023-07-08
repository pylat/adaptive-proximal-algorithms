include(joinpath(@__DIR__, "..", "counting.jl"))
include(joinpath(@__DIR__, "..", "recording.jl"))

using LinearAlgebra
using ProximalCore
using AdaProx
using Plots
using LaTeXStrings

struct WorstQuadratic{S, T}
    k::S
    L::T
end

function (f::WorstQuadratic)(x)
    s = x[1]^2 + x[f.k]^2
    for i in 1:(f.k - 1)
        s += (x[i] - x[i+1])^2
    end
    return (f.L / 4) * (s / 2 - x[1])
end

function ProximalCore.gradient!(grad, f::WorstQuadratic, x)
    grad[1] = (f.L / 4) * (2 * x[1] - x[2] - 1)
    for i in 2:(f.k - 1)
        grad[i] = (f.L / 4) * (2 * x[i] - x[i-1] - x[i+1])
    end
    grad[f.k] = (f.L / 4) * (2 * x[f.k] - x[f.k-1])
    grad[(f.k + 1):end] .= 0
    # since f is quadratic, f(x) = 1/2 <x, Q x> + <q, x>
    # meaning \nabla f(x) = Q x + q
    # and f(x) = (1/2) <\nabla f(x), x> + (1/2) <q, x>
    # since q = -(L/4) e_1, we obtain the following
    return dot(grad, x) / 2 - (f.L / 8) * x[1]
end

function main()
    k = 100
    n = 100

    @assert n >= k

    L = 10.0

    f = WorstQuadratic(100, 10.0)
    g = ProximalCore.Zero()

    optimum_value = (L / 8) * (1 / (k + 1) - 1)

    tol = 1e-6
    maxit = 10_000
    gam_init = 1 / L
    
    @info "Running solvers"

    sol, numit, record_fixed = AdaProx.fixed_proxgrad(
        zeros(n),
        f = Counting(f),
        g = g,
        gamma = gam_init,
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "PGM, fixed step 1/Lf"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit, record_backtracking = AdaProx.backtracking_proxgrad(
        zeros(n),
        f = Counting(f),
        g = g,
        gamma0 = 1.0,
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "PGM, backtracking step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit, record_backtracking_nesterov = AdaProx.backtracking_nesterov(
        zeros(n),
        f = Counting(f),
        g = g,
        gamma0 = 1.0,
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "Nesterov PGM, backtracking step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit, record_mm = AdaProx.adaptive_proxgrad(
        zeros(n),
        f = Counting(f),
        g = g,
        rule = AdaProx.MalitskyMishchenkoRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "PGM, MM adaptive step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit, record_our = AdaProx.adaptive_proxgrad(
        zeros(n),
        f = Counting(f),
        g = g,
        rule = AdaProx.OurRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "PGM, our adaptive step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    to_plot = Dict(
        "PGM (fixed step 1/L)" => concat_dicts(record_fixed),
        "PGM-ls" => concat_dicts(record_backtracking),
        "Nesterov-ls" => concat_dicts(record_backtracking_nesterov),
        "AdaPGM-MM" => concat_dicts(record_mm),
        "AdaPGM" => concat_dicts(record_our),
    )

    @info "Plotting"

    plot(
        title = "Lasso (random)",
        xlabel = "gradient evals f",
        ylabel = L"F(x^k) - F_\star",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:grad_f_evals],
            max.(1e-14, to_plot[k][:objective] .- optimum_value),
            yaxis = :log,
            label = k,
        )
    end
    savefig(joinpath(@__DIR__, "cost.pdf"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

