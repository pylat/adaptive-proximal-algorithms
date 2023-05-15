using Yota
using Zygote: pullback
using ReverseDiff
using ProximalCore

struct YotaFunction{F}
    f::F
end

(f::YotaFunction)(x) = f.f(x)

function ProximalCore.gradient!(grad, f::YotaFunction, x)
    f_x, g = Yota.grad(f.f, x)
    grad .= g[2]
    return f_x
end

struct ZygoteFunction{F}
    f::F
end

(f::ZygoteFunction)(x) = f.f(x)

function ProximalCore.gradient!(grad, f::ZygoteFunction, x)
    f_x, pb = pullback(f.f, x)
    grad .= pb(one(f_x))[1]
    return f_x
end

struct ReverseDiffFunction{F,T,X}
    f::F
    tape::T
end

function ReverseDiffFunction(f, x)
    f_tape = ReverseDiff.GradientTape(f, (x,))
    compiled_f_tape = ReverseDiff.compile(f_tape)
    return ReverseDiffFunction(f, compiled_f_tape)
end

(f::ReverseDiffFunction)(x) = f.f(x)

function ProximalCore.gradient!(grad, f::ReverseDiffFunction, x)
    ReverseDiff.gradient!((grad,), f.tape, (x,))
    # TODO can we get f.f(x) together with gradient?
    return f.f(x)
end
