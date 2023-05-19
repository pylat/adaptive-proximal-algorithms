record_grad_count(_) = nothing
record_grad_count(c::Counting) = c.count_gradient

record_prox_count(_) = nothing
record_prox_count(c::Counting) = c.count_prox

record_mul_count(_) = nothing
record_mul_count(c::Counting) = c.count_mul

record_amul_count(_) = nothing
record_amul_count(c::Counting) = c.count_amul

nocount(obj) = obj
nocount(c::Counting) = c.f

record_pg(x, f, g, gamma, norm_res) = Dict(
    :objective => nocount(f)(x) + nocount(g)(x),
    :grad_f_evals => record_grad_count(f),
    :prox_g_evals => record_prox_count(g),
    :gamma => gamma,
    :norm_res => norm_res,
)

record_pd(x, y, f, g, h, A, gamma, sigma, norm_res) = Dict(
    :objective => obj(f,g, h, A, x), 
    :grad_f_evals => record_grad_count(f),
    :prox_g_evals => record_prox_count(g),
    :prox_h_evals => record_prox_count(h),
    :A_evals => record_mul_count(A),
    :At_evals => record_amul_count(A),
    :gamma => gamma,
    :sigma => sigma,
    :norm_res => norm_res,
)

function obj(f, g, h, A, x) 
    y = try 
        nocount(f)(x) +
        nocount(g)(x) + 
        nocount(h)(nocount(A) * x)
    catch e 
        nocount(f)(x)
    end 
    return y
end


concat_dicts(dicts) = Dict(k => [d[k] for d in dicts] for k in keys(dicts[1]))

function subsample(n, collection)
    step = length(collection) / n |> ceil |> Int
    return collection[1:step:end]
end

subsample(n) = collection -> subsample(n, collection)
