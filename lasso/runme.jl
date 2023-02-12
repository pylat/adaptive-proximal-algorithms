include(joinpath(@__DIR__, "..", "counting.jl"))
include(joinpath(@__DIR__, "..", "recording.jl"))
include(joinpath(@__DIR__, "..", "adaptive_proximal_algorithms.jl"))

using Random
using LinearAlgebra
using DelimitedFiles
using Plots
using LaTeXStrings
using ProximalOperators: NormL1

pgfplotsx()

struct LinearLeastSquares{TA,Tb}
    A::TA
    b::Tb
end

(f::LinearLeastSquares)(w) = 0.5 * norm(f.A * w - f.b)^2

function gradient(f::LinearLeastSquares, w)
    res = f.A * w - f.b
    g = f.A' * res
    return g, 0.5 * norm(res)^2
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
    @info "Start Lasso"

    #= implementing 1/2 \|Ax-b\|^2 + λ/2 \|x\|_1
    		 =#
    Random.seed!(seed)

    T = Float64
    I = Int64

    p = n / pfactor # nonzeros
    rho = 1 # some positive number controlling how large solution is
    lam = 1  # check if we can change this without zeroing the solution---------<<<<<

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


    Lf = opnorm(A)^2
    gam_init = 1 / Lf
    f = LinearLeastSquares(A, b)
    g = NormL1(lam)

    @info "Running solvers"

    sol, numit, record_fixed = fixed_proxgrad(
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

    sol, numit, record_backtracking = backtracking_proxgrad(
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

    sol, numit, record_backtracking_nesterov = backtracking_nesterov(
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

    sol, numit, record_mm = adaptive_proxgrad(
        zeros(n),
        f = Counting(f),
        g = g,
        rule = MalitskyMischenkoRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "PGM, MM adaptive step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit, record_our = adaptive_proxgrad(
        zeros(n),
        f = Counting(f),
        g = g,
        rule = OurRule(gamma = gam_init),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "PGM, our adaptive step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    @info "Running aGRAAL"
    sol, numit, record_agraal = agraal(
        zeros(n),
        f = Counting(f),
        g = g,
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    @info "Collecting plot data"

    to_plot = Dict(
        "PGM (fixed step 1/L)" => concat_dicts(record_fixed),
        "PGM-ls" => concat_dicts(record_backtracking),
        "Nesterov-ls" => concat_dicts(record_backtracking_nesterov),
        "AdaPGM-MM" => concat_dicts(record_mm),
        "AdaPGM" => concat_dicts(record_our),
        "aGRAAL" => concat_dicts(record_agraal),
    )

    @info "Plotting"

    plot(
        title = "Lasso (random)",
        xlabel = L"\nabla f\ \mbox{evaluations}",
        ylabel = L"F(x^k) - F_\star",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:grad_f_evals],
            max.(1e-14, to_plot[k][:objective] .- optimum),
            yaxis = :log,
            label = k,
        )
    end
    savefig(joinpath(
        @__DIR__, 
        "convergence-$(size(A, 1))-$(size(A, 2))-$(Int(ceil(lam * 100)))-$(Int(p)).pdf"
    ))

    plot(
        title = "Lasso (random)",
        xlabel = L"\nabla f\ \mbox{evaluations}",
        ylabel = L"gamma",
    )
    for k in keys(to_plot)
        plot!(to_plot[k][:grad_f_evals], to_plot[k][:gamma], label = k)
    end
    savefig(joinpath(
        @__DIR__, 
        "gamma-$(size(A, 1))-$(size(A, 2))-$(Int(ceil(lam * 100)))-$(Int(p)).pdf"
    ))

    @info "Exporting plot data"

    save_labels = Dict(
        "PGM (fixed step 1/L)" => "PGM_fixed",
        "PGM-ls" => "PGM_bt",
        "Nesterov-ls" => "Nesterov_bt",
        "AdaPGM-MM" => "PGM_MM",
        "AdaPGM" => "PGM_our",
        "aGRAAL" => "aGraal",
    )

    for k in keys(to_plot)
        d = length(to_plot[k][:grad_f_evals])
        rr = Int(ceil(d / 50)) # keeping at most 50 data points
        output = [to_plot[k][:grad_f_evals] max.(1e-14, to_plot[k][:objective] .- optimum)]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$(size(A, 1))-$(size(A, 2))-$(Int(ceil(lam * 100)))-$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "convergence_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end

    for k in keys(to_plot)
        d = length(to_plot[k][:grad_f_evals])
        rr = Int(ceil(d / 200)) # keeping at most 50 data points
        output = [to_plot[k][:grad_f_evals] to_plot[k][:gamma]]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$(size(A, 1))-$(size(A, 2))-$(Int(ceil(lam * 100)))-$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "gamma_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end
end

function main()
    col = [
        (100, 300, 10),
        (500, 1000, 10),
        (4000, 1000, 10),
    ]
    for (m, n, pf) in col
        run_random_lasso(
            m = m,
            n = n,
            pfactor = pf,
            maxit = 2000,
            tol = 1e-7,
            seed = 0,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
