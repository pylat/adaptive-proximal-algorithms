include(joinpath(@__DIR__, "..", "libsvm.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))


using LinearAlgebra
using SparseArrays
using Logging: with_logger
using Tables
using DataFrames
using AdaProx
using Random
using Plots
using LaTeXStrings
using ProximalCore
using ProximalOperators: IndBox, IndZero

pgfplotsx()

struct Quadratic{TQ,Tq}
    Q::TQ
    q::Tq
end

function (f::Quadratic)(x)
    temp = f.Q * x
    return 0.5 * dot(x, temp) + dot(x, f.q)
end

function ProximalCore.gradient!(grad, f::Quadratic, x)
    temp = f.Q * x
    grad .= temp + f.q
    return 0.5 * dot(x, temp) + dot(x, f.q)
end

function run_dsvm(
    filename,
    ::Type{T} = Float64;
    tol = 1e-5,
    maxit = 1000,
    C = 1e-1,
    seed = 0,
    t = 1.0,
) where {T}
    @info "Start dual SVM ($filename)"

    Random.seed!(seed)
    X, y = load_libsvm_dataset(filename, T, labels = [-1.0, 1.0])

    m, n = size(X)
    N = size(y, 1)

    Dy = diagm(0 => y)
    Q = Dy * X * X' * Dy
    q = -ones(N)

    f = Quadratic(Q, q)
    g = IndBox(0.0, C)
    h = IndZero()
    A = y'

    Lf = norm(Q)
    x0 = zeros(N,1)
    y0 = zeros(1,1)
    norm_A = norm(A)

    t_values = [0.01, 0.15, 0.02, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    for t in t_values
        solx, soly, num_it = AdaProx.adaptive_primal_dual(
            x0,
            y0;
            f = AdaProx.Counting(f),
            g = g,
            h = h,
            A = A,
            rule = AdaProx.OurRule(t = t, norm_A = norm(A)),
            maxit = maxit,
            tol = tol,
            name = "AdaPDM (t=$t)",
        )
    end

    for t in t_values
        solx, soly, num_it = AdaProx.malitsky_pock(
            x0,
            y0;
            f = AdaProx.Counting(f),
            g = g,
            h = h,
            A = A,
            t = t,
            sigma = 1/norm_A,
            maxit = maxit,
            tol = tol,
            name = "Malitsky-Pock (t=$t)",
        )
    end 

    solx, soly, num_it = AdaProx.condat_vu(
        x0,
        y0;
        f = AdaProx.Counting(f),
        g = g,
        h = h,
        A = A,
        Lf = Lf,
        maxit = maxit,
        tol = tol,
        name = "Condat-Vu",
    )
end

function plot_residual(path)
    df = eachline(path) .|> JSON.parse |> Tables.dictrowtable |> DataFrame
    gb = groupby(df, :method)

    names_to_plot = []
    for name in ["Condat-Vu", "Malitsky-Pock", "AdaPDM"]
        matching_names = [k for k in keys(gb) if startswith(k.method, name)]
        push!(names_to_plot, find_best(gb, matching_names, :norm_res, 1e-5, :grad_f_evals))
    end

    fig = plot(
        title = "Dual SVM ($(basename(path)))",
        xlabel = "#passes through data",
        ylabel = L"\|v\|",
    )

    for k in names_to_plot
        if k.method === nothing
            continue
        end
        plot!(
            gb[k][!, :grad_f_evals],
            gb[k][!, :norm_res],
            yaxis = :log,
            label = k.method,
        )
    end

    savefig(fig, joinpath(@__DIR__, "$(basename(path)).pdf"))
end


function main(;maxit = 10_000)
    keys_to_log = [:method, :it, :grad_f_evals, :norm_res]

    for C  in [0.1, 1]
        path = joinpath(@__DIR__, "svmguide3_C_$(C).jsonl")
        with_logger(get_logger(path, keys_to_log)) do
            run_dsvm(
                joinpath(@__DIR__, "../", "datasets", "svmguide3"),
                maxit = maxit,
                tol = 1e-5,
                C = C
            )
        end
        plot_residual(path)

        path = joinpath(@__DIR__, "mushrooms_C_$(C).jsonl")
        with_logger(get_logger(path, keys_to_log)) do
            run_dsvm(
                joinpath(@__DIR__, "../", "datasets", "mushrooms"),
                maxit = maxit,
                tol = 1e-5,
                C = C
            )
        end
        plot_residual(path)

        path = joinpath(@__DIR__, "heart_scale_C_$(C).jsonl")
        with_logger(get_logger(path, keys_to_log)) do
            run_dsvm(
                joinpath(@__DIR__, "../", "datasets", "heart_scale"),
                maxit = maxit,
                tol = 1e-5,
                C = C
            )
        end
        plot_residual(path)
    end 
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
