include(joinpath(@__DIR__, "..", "counting.jl"))
include(joinpath(@__DIR__, "..", "recording.jl"))
include(joinpath(@__DIR__, "..", "adaptive_proximal_algorithms.jl"))
include(joinpath(@__DIR__, "..", "libsvm.jl"))

using Random
using LinearAlgebra
using Statistics
using DelimitedFiles
using Plots
using LaTeXStrings
using ProximalOperators: NormL1

pgfplotsx()

sigm(z) = 1 / (1 + exp(-z))

struct LogisticLoss{TX,Ty}
    X::TX
    y::Ty
end

function (f::LogisticLoss)(w)
    probs = sigm.(f.X * w[1:end-1] .+ w[end])
    return -mean(f.y .* log.(probs) + (1 .- f.y) .* log.(1 .- probs))
end

function gradient(f::LogisticLoss, w)
    probs = sigm.(f.X * w[1:end-1] .+ w[end])
    N = size(f.y, 1)
    g = f.X' * (probs - f.y) ./ N
    push!(g, mean(probs - f.y))  # for bias: X_new = [X, 1] 
    return g, f(w)
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

    Lf = norm(X * X') / 4 # doesn't take the column of 1s into account!
    gam_init = 1 / Lf

    @info "Getting accurate solution"

    sol, numit, _ = adaptive_proxgrad(
        zeros(n),
        f = f,
        g = g,
        rule = OurRule(gamma = 1.0),
        tol = tol / 10,
        maxit = maxit * 10,
    )
    optimum = f(sol) + g(sol)

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
        maxit = maxit/2,
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
        maxit = maxit/2,
        record_fn = record_pg,
    )
    @info "Nesterov PGM, backtracking step"
    @info "    iterations: $(numit)"
    @info "     objective: $(f(sol) + g(sol))"

    sol, numit, record_mm = adaptive_proxgrad(
        zeros(n),
        f = Counting(f),
        g = g,
        rule = MalitskyMishchenkoRule(gamma = 1.0),
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
        rule = OurRule(gamma = 1.0),
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
        title = "Logistic regression ($(basename(filename)))",
        xlabel = L"\nabla f\ \mbox{evaluations}",
        ylabel = L"F(x^k) - F_\star",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:grad_f_evals],
            max.(1e-14, to_plot[k][:objective] .- optimum),
            # max.(1e-14, to_plot[k][:norm_res]),
            yaxis = :log,
            label = k,
        )
    end
    savefig(joinpath(
        @__DIR__,
        "convergence_logreg_l1_$(basename(filename)).pdf"
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

    p = 1.0 # potential identifier

    for k in keys(to_plot)
        d = length(to_plot[k][:grad_f_evals])
        rr = Int(ceil(d / 80)) # keeping at most 50 data points
        output = [to_plot[k][:grad_f_evals] max.(1e-14, to_plot[k][:objective] .- optimum)]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$m-$n-$(Int(ceil(lam)))-$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "convergence_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end

    for k in keys(to_plot)
        d = length(to_plot[k][:grad_f_evals])
        rr = Int(ceil(d / d)) # keeping at most 50 data points
        output = [1:d to_plot[k][:gamma]]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$m-$n-$(Int(ceil(lam)))-$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "gamma_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end
    
end

function main()
    run_logreg_l1_data(
        joinpath(@__DIR__, "..", "datasets", "mushrooms"),
        lam = 0.01, maxit = 2000, tol = 1e-7
    )
    run_logreg_l1_data(
        joinpath(@__DIR__, "..", "datasets", "a5a"),
        lam = 0.01, maxit = 2000, tol = 1e-7
    )
    run_logreg_l1_data(
        joinpath(@__DIR__, "..", "datasets", "phishing"),
        lam = 0.01, maxit = 2000, tol = 1e-7
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
