module AdaProx

using Logging
using LinearAlgebra
using ProximalCore: prox, gradient, convex_conjugate, Zero

const Record = Logging.LogLevel(-1)

# Utilities.

include("./counting.jl")

is_logstep(n; base = 10) = mod(n, base^(log(base, n) |> floor)) == 0

nan_to_zero(v) = ifelse(isnan(v), zero(v), v)

upper_bound(x, f_x, grad_x, z, gamma) = f_x + real(dot(grad_x, z - x)) + 1 / (2 * gamma) * norm(z - x)^2

# Proximal-gradient methods with backtracking stepsize ("sufficient descent").
#
# See sections 10.4.2 and 10.7 from Amir Beck, "First-Order Methods in Optimization,"
# MOS-SIAM Series on Optimization, SIAM, 2017.
# https://my.siam.org/Store/Product/viewproduct/?ProductId=29044686

function backtrack_stepsize(gamma, f, g, x, f_x, grad_x)
    z, _ = prox(g, x - gamma * grad_x, gamma)
    ub_z = upper_bound(x, f_x, grad_x, z, gamma)
    f_z = f(z)
    while f_z > ub_z
        gamma /= 2
        if gamma < 1e-12
            @error "step size became too small ($gamma)"
        end
        z, _ = prox(g, x - gamma * grad_x, gamma)
        ub_z = upper_bound(x, f_x, grad_x, z, gamma)
        f_z = f(z)
    end
    return gamma, z, f_z
end

function backtracking_proxgrad(x0; f, g, gamma0, xi = 1.0 ,tol = 1e-5, maxit = 100_000, name = "Backtracking PG")
    x, z, gamma = x0, x0, gamma0
    grad_x, f_x = gradient(f, x)
    for it = 1:maxit
        gamma, z, f_z = backtrack_stepsize(xi * gamma, f, g, x, f_x, grad_x)
        norm_res = norm(z - x) / gamma
        @logmsg Record "" method=name it gamma norm_res objective=(nocount(f)(x) + nocount(g)(x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        if norm_res <= tol
            return z, it
        end
        x, f_x = z, f_z
        grad_x, _ = gradient(f, x)
    end
    return z, maxit
end

function backtracking_nesterov(x0; f, g, gamma0, tol = 1e-5, maxit = 100_000, name = "Backtracking Nesterov")
    x, z, gamma = x0, x0, gamma0
    theta = one(gamma)
    grad_x, f_x = gradient(f, x)
    for it = 1:maxit
        z_prev = z
        gamma, z, _ = backtrack_stepsize(gamma, f, g, x, f_x, grad_x)
        norm_res = norm(z - x) / gamma
        @logmsg Record "" method=name it gamma norm_res objective=(nocount(f)(x) + nocount(g)(x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        if norm_res <= tol
            return z, it
        end
        theta_prev = theta
        theta = (1 + sqrt(1 + 4 * theta_prev^2)) / 2
        x = z + (theta_prev - 1) / theta * (z - z_prev)
        grad_x, f_x = gradient(f, x)
    end
    return z, maxit
end

# Fixed stepsize fast proximal gradient
#
# See Chambolle, Pock, "An introduction to continuous optimization for imaging," 
# Acta Numerica, 25 (2016), 161–319.

function fixed_nesterov(
    x0;
    f,
    g,
    Lf = nothing,
    muf = 0,
    mug = 0,
    gamma = nothing,
    theta = nothing,
    tol = 1e-5,
    maxit = 100_000,
    name = "Fixed Nesterov"
)
    @assert (gamma === nothing) != (Lf === nothing)
    if gamma === nothing
        gamma = 1 / Lf
    end
    mu = muf + mug
    q = gamma * mu / (1 + gamma * mug)
    @assert q < 1
    if theta === nothing
        theta = if q > 0
            1 / sqrt(q)
        else
            0
        end
    end
    @assert 0 <= theta <= 1 / sqrt(q)
    x, x_prev = x0, x0
    for it = 1:maxit
        theta_prev = theta
        if mu == 0
            theta = (1 + sqrt(1 + 4 * theta_prev^2)) / 2
            beta = (theta_prev - 1) / theta
        else
            theta = (1 - q * theta_prev^2 + sqrt((1 - q * theta_prev^2)^2 + 4 * theta_prev^2)) / 2
            beta = (theta_prev - 1) * (1 + gamma * mug - theta * gamma * mu) / theta / (1 - gamma * muf)
        end
        z = x + beta * (x - x_prev)
        grad_z, _ = gradient(f, z)
        x_prev = x
        x, _ = prox(g, z - gamma * grad_z, gamma)
        norm_res = norm(x - z) / gamma
        @logmsg Record "" method=name it gamma norm_res objective=(nocount(f)(x) + nocount(g)(x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        if norm_res <= tol
            return x, it
        end
    end
    return x, maxit
end

# Adaptive Golden Ratio Algorithm.
#
# See Yura Malitsky, "Golden ratio algorithms for variational inequalities,"
# Mathematical Programming, Volume 184, Pages 383–410, 2020.
# https://link.springer.com/article/10.1007/s10107-019-01416-w

function agraal(
    x1;
    f,
    g,
    x0 = nothing,
    gamma0 = nothing,
    gamma_max = 1e6,
    phi = 1.5,
    tol = 1e-5,
    maxit = 100_000,
    name = "aGRAAL"
)
    if x0 === nothing
        x0 = x1 + randn(size(x1))
    end
    x, x_prev, x_bar = x1, x0, x1
    grad_x, _ = gradient(f, x)
    grad_x_prev, _ = gradient(f, x_prev)
    if gamma0 === nothing
        gamma0 = norm(x - x_prev) / norm(grad_x - grad_x_prev)
    end
    gamma = gamma0
    rho = 1 / phi + 1 / phi^2
    theta = one(gamma)
    for it = 1:maxit
        C = norm(x - x_prev)^2 / norm(grad_x - grad_x_prev)^2
        gamma_prev = gamma
        gamma = min(rho * gamma_prev, phi * theta * C / (4 * gamma_prev), gamma_max)
        theta = phi * gamma / gamma_prev
        x_bar = ((phi - 1) * x + x_bar) / phi
        x_prev, grad_x_prev = x, grad_x
        x, _ = prox(g, x_bar - gamma * grad_x_prev, gamma)
        norm_res = norm(x - x_prev) / gamma
        @logmsg Record "" method=name it gamma norm_res objective=(nocount(f)(x) + nocount(g)(x)) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) f_evals=eval_count(f)
        if norm_res <= tol
            return x, it
        end
        grad_x, _ = gradient(f, x)
    end
    return x, maxit
end

# Fixed-step and adaptive primal-dual and proximal-gradient methods.
# All algorithms implemented as special cases of one generic loop.
#
# See:
# - Chapter 10 from Amir Beck, "First-Order Methods in Optimization,"
#   MOS-SIAM Series on Optimization, SIAM, 2017.
#   https://my.siam.org/Store/Product/viewproduct/?ProductId=29044686
# - Yura Malitsky, Konstantin Mishchenko "Adaptive Gradient Descent without Descent,"
#   Proceedings of the 37th International Conference on Machine Learning, PMLR 119:6702-6712, 2020.
#   https://proceedings.mlr.press/v119/malitsky20a.html
# - Laurent Condat, "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms,"
#   Journal of optimization theory and applications, Springer, 2013.
#   https://link.springer.com/article/10.1007/s10957-012-0245-9

Base.@kwdef struct FixedStepsize{R}
    gamma::R
    t::R = one(gamma)
end

function stepsize(rule::FixedStepsize, args...)
    return (rule.gamma, rule.gamma * rule.t^2), nothing
end

Base.@kwdef struct MalitskyMishchenkoRule{R}
    gamma::R
    t::R = one(gamma)
end

function stepsize(rule::MalitskyMishchenkoRule{R}) where {R}
    return (rule.gamma, rule.gamma * rule.t^2), (rule.gamma, R(Inf))
end

function stepsize(rule::MalitskyMishchenkoRule, (gamma_prev, rho), x1, grad_x1, x0, grad_x0)
    L = norm(grad_x1 - grad_x0) / norm(x1 - x0)
    gamma = min(sqrt(1 + rho) * gamma_prev, 1 / (2 * L))
    return (gamma, gamma * rule.t^2), (gamma, gamma / gamma_prev)
end

struct OurRule{R}
    gamma::R
    t::R
    norm_A::R
    delta::R
    Theta::R
end

function OurRule(; gamma = 0, t = 1, norm_A = 0, delta = 0, Theta = 1.2)
    _gamma = if gamma > 0
        gamma
    elseif norm_A > 0
        1 / (2 * Theta * t * norm_A)
    else
        error("you must provide gamma > 0 if norm_A = 0")
    end
    R = typeof(_gamma)
    return OurRule{R}(_gamma, R(t), R(norm_A), R(delta), R(Theta))
end

function stepsize(rule::OurRule)
    gamma = rule.gamma
    sigma = rule.gamma * rule.t^2
    return (gamma, sigma), (gamma, gamma)
end

function stepsize(rule::OurRule, (gamma1, gamma0), x1, grad_x1, x0, grad_x0)
    xi = rule.t^2 * gamma1^2 * rule.norm_A^2
    C = norm(grad_x1 - grad_x0)^2 / dot(grad_x1 - grad_x0, x1 - x0) |> nan_to_zero
    L = dot(grad_x1 - grad_x0, x1 - x0) / norm(x1 - x0)^2 |> nan_to_zero
    D = gamma1 * L * (gamma1 * C - 1)
    gamma = min(
        gamma1 * sqrt(1 + gamma1 / gamma0),
        1 / (2 * rule.Theta * rule.t * rule.norm_A),
        (
            gamma1 * sqrt(1 - 4 * xi * (1 + rule.delta)^2) /
            sqrt(2 * (1 + rule.delta) * (D + sqrt(D^2 + xi * (1 - 4 * xi * (1 + rule.delta)^2))))
        ),
    )
    sigma = gamma * rule.t^2
    return (gamma, sigma), (gamma, gamma1)
end

function adaptive_primal_dual(
    x,
    y;
    f,
    g,
    h,
    A,
    rule,
    tol = 1e-5,
    maxit = 10_000,
    name = "AdaPDM",
)
    (gamma, sigma), state = stepsize(rule)
    h_conj = convex_conjugate(h)

    A_x = A * x
    grad_x, _ = gradient(f, x)
    At_y = A' * y
    v = x - gamma * (grad_x + At_y)
    x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
    x, _ = prox(g, v, gamma)

    for it = 1:maxit
        A_x = A * x
        grad_x, _ = gradient(f, x)

        primal_res = (v - x) / gamma + grad_x + At_y

        gamma_prev = gamma
        (gamma, sigma), state = stepsize(rule, state, x, grad_x, x_prev, grad_x_prev)
        rho = gamma / gamma_prev

        w = y + sigma * ((1 + rho) * A_x - rho * A_x_prev)
        y, _ = prox(h_conj, w, sigma)

        dual_res = (w - y) / sigma - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        @logmsg Record "" method=name it gamma sigma norm_res objective=obj(f, g, h, A, x) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        if norm_res <= tol
            return x, y, it
        end

        At_y = A' * y
        v = x - gamma * (grad_x + At_y)
        x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
        x, _ = prox(g, v, gamma)
    end
    return x, y, maxit
end


function condat_vu(
    x,
    y;
    f,
    g,
    h,
    A,
    Lf,
    gamma = nothing,
    sigma = nothing,
    norm_A = nothing,
    tol = 1e-5,
    maxit = 10_000,
    name = "Condat-Vu",
)
    # # NOTE: Peviously I had parameter selection as per [Thm 3.1, Condat 2013]
    # # Implemented as follows (rho is relaxation parameter)
    #     if gamma === nothing && sigma !== nothing
    #         gamma = 0.99 / (Lf / 2 + sigma * norm_A^2)
    #     elseif gamma !== nothing && sigma === nothing
    #         sigma = 0.99 * (1 / gamma - Lf / 2) / norm_A^2
    #     end
    #     @assert gamma !== nothing && sigma !== nothing
    #     if rho === nothing
    #         delta = 2 - Lf / (2 * (1 / gamma - sigma * norm_A^2))
    #         rho = delta / 2
    #     end
    #     gamma_sigma = 1 / gamma - sigma * norm_A^2
    #     @assert gamma_sigma >= Lf / 2
    #     @assert (rho > 0) && (rho < 2 - Lf / 2 / gamma_sigma)

    if gamma === sigma === nothing
        R = typeof(Lf)
        par = R(5) # scaling parameter for comparing Lipschitz constants and \|L\|
        par2 = R(100)   # scaling parameter for α
        if norm_A === nothing
            norm_A = norm(A)
        end
        if norm_A > par * Lf
            alpha = R(1)
        else
            alpha = par2 * norm_A / Lf
        end
        gamma = R(1) / (Lf / 2 + norm_A / alpha)
        sigma = R(0.99) / (norm_A * alpha)
    end
    @assert gamma !== nothing && sigma !== nothing
    rule = FixedStepsize(gamma, sqrt(sigma / gamma))
    return adaptive_primal_dual(x, y; f, g, h, A, rule, tol, maxit, name)
end

function adaptive_proxgrad(x; f, g, rule, tol = 1e-5, maxit = 100_000, name = "AdaPGM")
    x, _, numit = adaptive_primal_dual(x, zero(x); f, g, h = Zero(), A = 0, rule, tol, maxit, name)
    return x, numit
end

function auto_adaptive_proxgrad(x; f, g, gamma = nothing, tol = 1e-5, maxit = 100_000, name = "AutoAdaPGM")
    grad_x, _ = gradient(f, x)

    if norm(grad_x) <= tol
        return x, 0
    end

    if gamma === nothing 
        xeps = prox(x .- 0.1 * grad_x, 0.1) # proxgrad
        grad_xeps, _ = gradient(f, xeps)
        L = dot(grad_x - grad_xeps, x - xeps) / norm(x - xeps)^2
        gamma = iszero(L) ? 1.0 : 1 / L  
    end 

    @assert gamma > 0
    
    x_prev, grad_x_prev, gamma_prev = x, grad_x, gamma
    x, _ = prox(g, x - gamma * grad_x, gamma)
    grad_x, _ = gradient(f, x)
    L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2
    gamma = iszero(L) ? sqrt(2) * gamma : 1 / L

    if gamma_prev / gamma > 1e5  # if the inital guess was too large
        x, _ = prox(g, x_prev - gamma * grad_x_prev, gamma)
        grad_x, _ = gradient(f, x)
        L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2
        gamma = iszero(L) ? sqrt(2) * gamma : 1 / L
    end     

    rule = OurRule(; gamma, t=1, norm_A=0, delta=0, Theta=1.2)

    return adaptive_proxgrad(x_prev; f, g, rule, tol, maxit, name = name)
end

function fixed_proxgrad(x; f, g, gamma, tol = 1e-5, maxit = 100_000, name = "Fixed stepsize PGM")
    adaptive_proxgrad(x; f, g, rule = FixedStepsize(gamma, one(gamma)), tol, maxit, name)
end

# Linesearch version of adaptive_primal_dual ("fully adaptive")

function adaptive_linesearch_primal_dual(
    x,
    y;
    f,
    g,
    h,
    A,
    gamma = nothing,
    eta = 1.0,
    t = 1.0,
    delta = 1e-8,
    Theta = 1.2,
    r = 2,
    R = 0.95,
    tol = 1e-5,
    maxit = 10_000,
    name = "AdaPDM+",
)
    @assert eta > 0 "eta must be positive"
    @assert Theta > (delta + 1) "must be Theta > (delta + 1)"

    if gamma === nothing
        gamma = 1 / (2 * Theta * t * eta)
    end

    @assert gamma <= 1 / (2 * Theta * t * eta) "gamma is too large"

    delta1 = 1 + delta
    gamma_prev = gamma
    h_conj = convex_conjugate(h)

    A_x = A * x
    grad_x, _ = gradient(f, x)
    At_y = A' * y
    v = x - gamma * (grad_x + At_y)
    x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
    x, _ = prox(g, v, gamma)

    for it = 1:maxit
        A_x = A * x
        grad_x, _ = gradient(f, x)

        primal_res = (v - x) / gamma + grad_x + At_y

        C = norm(grad_x - grad_x_prev)^2 / dot(grad_x - grad_x_prev, x - x_prev) |> nan_to_zero
        L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2 |> nan_to_zero
        Delta = gamma * L * (gamma * C - 1)
        xi_bar = t^2 * gamma^2 * eta^2 * delta1^2
        m4xim1 = (1 - 4 * xi_bar)

        eta = R * eta
        w = y
        sigma = t^2 * gamma
        while true
            gamma_next = min(
                gamma * sqrt(1 + gamma / gamma_prev),
                1 / (2 * Theta * t * eta),
                gamma * sqrt(m4xim1 / (2 * delta1 * (Delta + sqrt(Delta^2 + m4xim1 * (t * eta * gamma)^2)))),
            )
            rho = gamma_next / gamma
            sigma = t^2 * gamma_next
            w = y + sigma * ((1 + rho) * A_x - rho * A_x_prev)
            y_next, _ = prox(h_conj, w, sigma)
            At_y_next = A' * y_next
            if eta >= norm(At_y_next - At_y) / norm(y_next - y)
                gamma, gamma_prev = gamma_next, gamma
                y, At_y = y_next, At_y_next
                break
            end
            eta *= r
        end

        dual_res = (w - y) / sigma - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        @logmsg Record "" method=name it gamma sigma norm_res objective=obj(f, g, h, A, x) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        if norm_res <= tol
            return x, y, it
        end

        v = x - gamma * (grad_x + At_y)
        x_prev, A_x_prev, grad_x_prev = x, A_x, grad_x
        x, _ = prox(g, v, gamma)
    end
    return x, y, maxit
end      
  
  
# Algorithm 4 of ``A first-order primal-dual algorithm with linesearch''
# (applied to the dual for consistency)

function backtrack_stepsize_MP(sigma, sigma_prev, t, x_prev, y, y_prev, grad_x_prev, A_x_prev, At_y, At_y_prev, f, g, A, f_x_prev)
    theta = sigma / sigma_prev
    gamma = t^2 * sigma
    At_ybar = (1+theta) * At_y - theta* At_y_prev
    v = x_prev - gamma * (At_ybar + grad_x_prev)
    x, _ = prox(g, v, gamma)
    A_x = A * x
    lhs = gamma * sigma * norm(A_x - A_x_prev)^2 + 2 * gamma * (f(x) - f_x_prev - dot(grad_x_prev, x - x_prev))
    while lhs > 0.95 * norm(x - x_prev)^2
        sigma /= 2
        if sigma < 1e-12
            @error "step size became too small ($gamma)"
        end
        theta = sigma / sigma_prev
        gamma = t^2 * sigma
        At_ybar = (1+theta) * At_y - theta* At_y_prev
        v = x_prev - gamma * (At_ybar + grad_x_prev)
        x, _ = prox(g, v, gamma)
        A_x = A * x
        lhs = gamma * sigma * norm(A_x - A_x_prev)^2 + 2 * gamma * (f(x) - f_x_prev - dot(grad_x_prev, x - x_prev))
    end
    return sigma, gamma, x, v, A_x
end

function malitsky_pock(
    x,
    y;
    f,
    g,
    h,
    A,
    sigma,
    t = 1.0, # t = gamma / sigma > 0
    tol = 1e-5,
    maxit = 10_000,
    name = "MP-ls",
)
    h_conj = convex_conjugate(h)
    theta = one(sigma)
    y_prev = y
    A_x = A * x
    At_y = A' * y
    for it = 1:maxit
        At_y_prev = At_y 
        w = y + sigma * A_x
        y, _ = prox(h_conj, w, sigma)
        At_y = A' * y

        sigma_prev = sigma
        sigma = sigma * sqrt(1 + theta)

        grad_x_prev, f_x_prev = gradient(f, x)
        x_prev, A_x_prev = x, A_x
        sigma, gamma, x, v, A_x =
            backtrack_stepsize_MP(sigma, sigma_prev, t, x_prev, y, y_prev, grad_x_prev, A_x_prev, At_y, At_y_prev, f, g, A, f_x_prev)
        grad_x, f_x = gradient(f, x)

        y_prev = y

        primal_res = (v - x) / gamma + grad_x + At_y
        dual_res = (w - y) / sigma_prev - A_x
        norm_res = sqrt(norm(primal_res)^2 + norm(dual_res)^2)

        @logmsg Record "" method=name it gamma sigma norm_res objective=obj(f, g, h, A, x) grad_f_evals=grad_count(f) prox_g_evals=prox_count(g) prox_h_evals=prox_count(h) A_evals=mul_count(A) At_evals=amul_count(A) f_evals=eval_count(f)
        if norm_res <= tol
            return x, y, it
        end
    end
    return x, y, maxit
end

end
