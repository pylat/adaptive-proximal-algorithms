include(joinpath(@__DIR__, "..", "libsvm.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))

using LinearAlgebra
using SparseArrays
using Logging: with_logger
using Tables
using DataFrames
using Plots
using LaTeXStrings
using Random
using ProximalCore: Zero
using ProximalOperators: NormL1, NormL2, Translate
using AdaProx

pgfplotsx()

function run_least_absolute_deviation(
    filename,
    ::Type{T} = Float64;
    lambda = 1e-1,
    seed = 0,
    tol = 1e-5,
    maxit = 1000,
) where {T}
    @info "Start run_least_absolute_deviation ($filename)"

    Random.seed!(seed)

    X, y = load_libsvm_dataset(filename, T)

    m, n = size(X)

    f = Zero()
    g = NormL1(lambda)
    h = Translate(NormL1(), -y)
    A = hcat(Matrix(X), ones(m, 1))

    Lf = lambda

    norm_A = norm(A)

    solx, soly, numit = AdaProx.vu_condat(
        zeros(n + 1),
        zeros(m);
        f = f,
        g = g,
        h = h,
        A = AdaProx.Counting(A),
        Lf = Lf,
        norm_A,
        maxit = maxit,
        tol = tol,
        name = "Vu-Condat"
    )

    for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        solx, soly, numit = AdaProx.malitsky_pock(
            zeros(n + 1),
            zeros(m);
            f = f,
            g = g,
            h = h,
            A = AdaProx.Counting(A),
            sigma = 1.0,
            t = t,
            maxit = maxit,
            tol = tol,
            name = "Malitsky-Pock (t=$t)",
        )
    end

    for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        solx, soly, numit = AdaProx.adaptive_linesearch_primal_dual_2(
            zeros(n + 1),
            zeros(m);
            f = f,
            g = g,
            h = h,
            A = AdaProx.Counting(A),
            t = t,
            maxit = maxit,
            tol = tol,
            name = "AdaPDM+ (t=$t)",
        )
    end

    solx, soly, numit = AdaProx.auto_adaptive_linesearch_primal_dual_2(
        zeros(n + 1),
        zeros(m);
        f = f,
        g = g,
        h = h,
        A = AdaProx.Counting(A),
        gamma = 1.0,
        eta = norm(A),
        maxit = maxit,
        tol = tol,
        name = "AutoAdaPDM+",
    )
end

function find_best(gb, names, key, target)
    best_name, rest_names = Iterators.peel(names)
    best_length = -1
    best_val = gb[best_name][!, key][end]
    if best_val <= target
        best_length = size(gb[best_name])[1]
    end
    for name in rest_names
        length = size(gb[name])[1]
        val = gb[name][!, key][end]
        if best_length >= 0 && val <= target && length < best_length
            best_name = name
            best_length = length
        elseif best_length < 0 && val < best_val
            best_name = name
            best_val = val
        end
    end
    return best_name
end

function plot_residual(path)
    df = eachline(path) .|> JSON.parse |> Tables.dictrowtable |> DataFrame
    gb = groupby(df, :method)

    names_to_plot = []
    for name in ["Vu-Condat", "Malitsky-Pock", "AdaPDM+", "AutoAdaPDM+"]
        matching_names = [k for k in keys(gb) if startswith(k.method, name)]
        push!(names_to_plot, find_best(gb, matching_names, :norm_res, 1e-5))
    end

    fig = plot(
        title = "Least absolute deviation ($(basename(path)))",
        xlabel = "#passes through data",
        ylabel = L"\|v\|",
    )

    for k in names_to_plot
        if k.method === nothing
            continue
        end
        plot!(
            gb[k][!, :A_evals] + gb[k][!, :At_evals],
            gb[k][!, :norm_res],
            yaxis = :log,
            label = k.method,
        )
    end

    savefig(fig, joinpath(@__DIR__, "$(basename(path)).pdf"))
end

function main(; maxit = 10_000)
    keys_to_log = [:method, :norm_res, :A_evals, :At_evals]

    path = joinpath(@__DIR__, "cpusmall_scale.jsonl")
    with_logger(get_logger(path, keys_to_log)) do
        run_least_absolute_deviation(
            joinpath(@__DIR__, "../", "datasets", "cpusmall_scale"),
            maxit = maxit,
            tol = 1e-5,
            lambda = 1e1,
        )
    end
    plot_residual(path)

    path = joinpath(@__DIR__, "abalone.jsonl")
    with_logger(get_logger(path, keys_to_log)) do
        run_least_absolute_deviation(
            joinpath(@__DIR__, "../", "datasets", "abalone"),
            maxit = maxit,
            tol = 1e-5,
            lambda = 1e1,
        )
    end
    plot_residual(path)

    path = joinpath(@__DIR__, "housing_scale.jsonl")
    with_logger(get_logger(path, keys_to_log)) do
        run_least_absolute_deviation(
            joinpath(@__DIR__, "../", "datasets", "housing_scale"),
            maxit = maxit,
            tol = 1e-5,
            lambda = 1e1,
        )
    end
    plot_residual(path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
