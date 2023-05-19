include(joinpath(@__DIR__, "..", "autodiff.jl"))
include(joinpath(@__DIR__, "..", "counting.jl"))
include(joinpath(@__DIR__, "..", "recording.jl"))
include(joinpath(@__DIR__, "..", "adaptive_proximal_algorithms.jl"))
include(joinpath(@__DIR__, "..", "libsvm.jl"))


using LinearAlgebra
using SparseArrays
using ProximalAlgorithms: AFBA, VuCondat

using Random

using Plots
using LaTeXStrings

using DelimitedFiles
using ProximalOperators: IndBox, IndZero

struct Quadratic{TQ,Tq}
    Q::TQ
    q::Tq
end


function (f::Quadratic)(x)
    temp = f.Q * x
    return 0.5 * dot(x, temp) + dot(x, f.q)
end

function gradient(f::Quadratic, x)
    temp = f.Q * x
    return temp + f.q, 0.5 * dot(x, temp) + dot(x, f.q)
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
    # A = Counting(A)

    Lf = norm(Q)
    x0 = zeros(N, 1)
    y0 = zeros(1, 1)
    norm_A = norm(A)


    println("t is $t")

    @info "Running solvers"


    solx, soly, num_it, record_our = adaptive_primal_dual(
        x0,
        y0;
        f = Counting(f),
        g = g,
        h = h,
        A = A,
        rule = OurRule(t = t, norm_A = norm(A)),
        maxit = maxit,
        tol = tol,
        record_fn = record_pd,
    )
    println(
        "Adaptive PD: $num_it iterations, $(f(solx) + g(solx)) cost, feasibility $(A * solx)",
    )


    solx, soly, num_it, record_ourpiu = adaptive_linesearch_primal_dual(
        x0,
        y0;
        f = Counting(f),
        g = g,
        h = h,
        A = A,
        eta = 1.0 * norm_A,
        # c = 1.01, 
        t = t,
        maxit = maxit,
        tol = tol,
        record_fn = record_pd,
    )
    println(
        "Adaptive PD+: $num_it iterations, $(f(solx) + g(solx)) cost, feasibility $(A * solx)",
    )

    solx, soly, num_it, record_MP = malitsky_pock(
        x0,
        y0;
        f = Counting(f),
        g = g,
        h = h,
        A = A,
        t = t,
        sigma = 1e-2,
        maxit = maxit,
        tol = tol,
        record_fn = record_pd,
    )
    println(
        "MalitskyPock: $num_it iterations, $(f(solx) + g(solx)) cost, feasibility $(A * solx)",
    )


    solx, soly, num_it, record_Vu = vu_condat(
        x0,
        y0;
        f = Counting(f),
        g = g,
        h = h,
        A = A,
        Lf = Lf,
        maxit = maxit,
        tol = tol,
        record_fn = record_pd,
    )
    println(
        "VuCondat: $num_it iterations, $(f(solx) + g(solx)) cost, feasibility $(A * solx)",
    )

    @info "Collecting plot data"

    to_plot = Dict(
        "Vu-Condat" => concat_dicts(record_Vu),
        "APDHG" => concat_dicts(record_our),
        "MP" => concat_dicts(record_MP),
        "APDHG+" => concat_dicts(record_ourpiu),
    )

    @info "Exporting plot data"

    save_labels = Dict(
        "Vu-Condat" => "Vu-Condat",
        "APDHG" => "APDHG",
        "MP" => "MP",
        "APDHG+" => "AdaPD+",
    )
    p = Int(floor(t * 1000)) # potential identifier for later!
    lam = C

    for k in keys(to_plot)
        d = length(to_plot[k][:grad_f_evals])
        rr = Int(ceil(d / 100)) # keeping at most 50 data points
        output = [to_plot[k][:grad_f_evals] max.(1e-14, to_plot[k][:norm_res])]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])_$(m)_$(n)_$(Int(ceil(lam * 100)))_$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "convergence_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end
end



function main(; maxit = 1000)
    for t in [1], C in [0.1, 1]
        run_dsvm(
            joinpath(@__DIR__, "../", "datasets", "svmguide3"),
            maxit = maxit,
            tol = 1e-5,
            C = C,
            t = t,
        )
        run_dsvm(
            joinpath(@__DIR__, "../", "datasets", "mushrooms"),
            maxit = maxit,
            tol = 1e-5,
            C = C,
            t = t,
        )
        run_dsvm(
            joinpath(@__DIR__, "../", "datasets", "heart_scale"),
            maxit = maxit,
            tol = 1e-5,
            C = C,
            t = t,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
