include(joinpath(@__DIR__, "..", "autodiff.jl"))
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

struct Cubic{TQ,Tq,R}
    Q::TQ
    q::Tq
    c::R
end

(f::Cubic)(x) = dot(x, f.Q * x) / 2 + dot(x, f.q) + norm(x)^3 * f.c / 6

function gradient(f::Cubic, x)
    g = f.Q * x + f.q + (f.c * norm(x) / 2) * x
    return g, (dot(f.q, g) + dot(f.q, x)) / 2 + norm(x)^3 * f.c / 6
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
    f = ZygoteFunction(f)
    g = Zero()

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
        rule = MalitskyMishchenkoRule(gamma = 0.001),
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
        # "PGM (fixed step 1/L)" => concat_dicts(record_fixed),
        "PGM-ls" => concat_dicts(record_backtracking),
        "Nesterov-ls" => concat_dicts(record_backtracking_nesterov),
        "AdaPGM-MM" => concat_dicts(record_mm),
        "AdaPGM" => concat_dicts(record_our),
        "aGRAAL" => concat_dicts(record_agraal),
    )

    @info "Plotting"

    plot(
        title = "cubic regularization ($(basename(filename)))",
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
    savefig(joinpath(@__DIR__, "convergence_cubic_$(basename(filename)).pdf"))

    @info "Exporting plot data"

    save_labels = Dict(
        # "PGM (fixed step 1/L)" => "PGM_fixed",
        "PGM-ls" => "PGM_bt",
        "Nesterov-ls" => "Nesterov_bt",
        "AdaPGM-MM" => "PGM_MM",
        "AdaPGM" => "PGM_our",
        "aGRAAL" => "aGraal",
    )
    p = 1.0 # potential identifier
    for k in keys(to_plot)
        d = length(to_plot[k][:grad_f_evals])
        rr = Int(ceil(d / 80)) # keeping at most 80 data points
        output = [to_plot[k][:grad_f_evals] to_plot[k][:gamma]]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$m-$n-$(Int(ceil(lam*100)))-$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "convergence_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end
    for k in keys(to_plot)
        d = length(to_plot[k][:grad_f_evals])
        rr = Int(ceil(d / 50)) # keeping at most 50 data points
        output = [to_plot[k][:grad_f_evals] max.(1e-14, to_plot[k][:objective] .- optimum)]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$m-$n-$(Int(ceil(lam*100)))-$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "convergence_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end
end


function main()
    run_cubic_logreg_data(
        joinpath(@__DIR__, "..", "datasets", "mushrooms"),
        lam = 1,
        maxit = 100,
        tol = 1e-7,
    )
    run_cubic_logreg_data(
        joinpath(@__DIR__, "..", "datasets", "phishing"),
        lam = 1,
        maxit = 100,
        tol = 1e-7,
    )
    run_cubic_logreg_data(
        joinpath(@__DIR__, "..", "datasets", "a5a"),
        lam = 1,
        maxit = 100,
        tol = 1e-7,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
