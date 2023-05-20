include(joinpath(@__DIR__, "..", "autodiff.jl"))
include(joinpath(@__DIR__, "..", "counting.jl"))
include(joinpath(@__DIR__, "..", "recording.jl"))
include(joinpath(@__DIR__, "..", "libsvm.jl"))


using LinearAlgebra
using SparseArrays

using Plots
using LaTeXStrings
using DelimitedFiles

using Random 

using ProximalCore: Zero
using ProximalOperators: NormL1, NormL2, Translate
using AdaProx

pgfplotsx()

# square root lasso

function run_square_root_lasso(filename, ::Type{T} = Float64; lambda = 1e-1, tol = 1e-5, maxit = 1000, t= 1.0, seed = 0) where {T}
    @info "Start square root lasso ($filename)"

    Random.seed!(seed)
    X, y = load_libsvm_dataset(filename, T)

    m, n = size(X)

    f = Zero()
    g = NormL1(lambda)
    h = Translate(NormL2(), -y)
    A = hcat(Matrix(X), ones(m, 1))

    Lf = lambda

    norm_A = norm(A)

    solx, soly, numit, record_vc = AdaProx.vu_condat(
        zeros(n + 1),
        zeros(m);
        f = f,
        g = g,
        h = h,
        A = Counting(A),
        Lf = Lf,
        norm_A,
        maxit = maxit,
        tol = tol,
        record_fn = record_pd,
    )
    @info "Vu-Condat" numit (f(solx) + h(A * solx))

    solx, soly, numit, record_mp = AdaProx.malitsky_pock(
        zeros(n + 1),
        zeros(m);
        f = f,
        g = g,
        h = h,
        A = Counting(A),
        sigma = 1.0,
        t = t,
        maxit = maxit,
        tol = tol,
        record_fn = record_pd,
    )
    @info "Malitsky-Pock" numit (f(solx) + h(A * solx))

    # solx, soly, numit, record_apd = adaptive_primal_dual(
    #     zeros(n + 1),
    #     zeros(m);
    #     f = f,
    #     g = g,
    #     h = h,
    #     A = Counting(A),
    #     rule = OurRule(t = t, norm_A = norm_A),
    #     maxit = maxit,
    #     tol = tol,
    #     record_fn = record_pd,
    # )
    # @info "Adaptive PD" numit (f(solx) + h(A * solx))

    solx, soly, numit, record_fapd = AdaProx.adaptive_linesearch_primal_dual(
        zeros(n + 1),
        zeros(m);
        f = f,
        g = g,
        h = h,
        A = Counting(A),
        t = t,
        maxit = maxit,
        tol = tol,
        record_fn = record_pd,
    )
    @info "Fully Adaptive PD" numit (f(solx) + h(A * solx))

    @info "Collecting plot data"

    to_plot = Dict(
        "Vu-Condat" => concat_dicts(record_vc),
        "MP" => concat_dicts(record_mp),
        # "adaPD" => concat_dicts(record_apd),
        "adaPD+" => concat_dicts(record_fapd),
    )


    @info "exporting data"

    save_labels = Dict(
        "Vu-Condat" => "Vu-Condat", 
        # "adaPD"     => "adaPD",
        "MP"        => "MP",
        "adaPD+" => "AdaPD+",
        )
    p = Int(floor(t*1000)) # potential identifier for later!
    lam = lambda 

    for k in keys(to_plot)
        d = length(to_plot[k][:A_evals])
        rr = Int(ceil(d / 100)) # keeping at most 50 data points
        output = [(to_plot[k][:A_evals]+ to_plot[k][:At_evals]) max.(1e-14, to_plot[k][:norm_res])]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])_$(m)_$(n)_$(Int(ceil(lam * 100)))_$(Int(p)).txt"
        filepath = joinpath(@__DIR__, "convergence_plot_data", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end

    @info "Plotting"

    plot(
        title = "square root lasso ($(basename(filename))) with lam$(Int(floor(lambda*100))), t$(Int(floor(t*100)))",
        xlabel = "#passes through data",
        ylabel = L"\|v\|",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:A_evals] + to_plot[k][:At_evals],
            to_plot[k][:norm_res],
            yaxis = :log,
            label = k,
        )
    end
    savefig(
        joinpath(
            @__DIR__,
            "convergence_SRL_$(basename(filename))_lam$(Int(floor(lambda*100)))_t$(Int(floor(t*100))).pdf",
        )
    )
end


function main(;maxit = 5000)
    # t is the ratio between the primal and the dual stepsizes 
    for t in [1.0], lam in [1e-1, 1e1]
        run_square_root_lasso(joinpath(@__DIR__, "../", "datasets", "cpusmall_scale"), maxit = maxit, tol = 1e-5, t= t)
        run_square_root_lasso(joinpath(@__DIR__, "../", "datasets", "abalone"), maxit = maxit, tol = 1e-5, t = t)
        run_square_root_lasso(joinpath(@__DIR__, "../", "datasets", "housing_scale"), maxit = maxit, tol = 1e-5, t = t)
    end 
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
