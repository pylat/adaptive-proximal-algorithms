using Yota
using Zygote: pullback
using ReverseDiff
using ProximalCore: Zero

function gradient(f::Zero, x)
    return zero(x), f(x)
end

struct YotaFunction{F}
    f::F
end

(f::YotaFunction)(x) = f.f(x)

function gradient(f::YotaFunction, x)
    f_x, g = Yota.grad(f.f, x)
    return g[2], f_x
end

struct ZygoteFunction{F}
    f::F
end

(f::ZygoteFunction)(x) = f.f(x)

function gradient(f::ZygoteFunction, x)
    f_x, pb = pullback(f.f, x)
    return pb(one(f_x))[1], f_x
end

struct ReverseDiffFunction{F,T,X}
    f::F
    grad::X
    tape::T
end

function ReverseDiffFunction(f, x)
    f_tape = ReverseDiff.GradientTape(f, (x,))
    compiled_f_tape = ReverseDiff.compile(f_tape)
    return ReverseDiffFunction(f, similar(x), compiled_f_tape)
end

(f::ReverseDiffFunction)(x) = f.f(x)

function gradient(f::ReverseDiffFunction, x)
    ReverseDiff.gradient!((f.grad,), f.tape, (x,))
    # TODO can we get f.f(x) together with gradient?
    return f.grad, f.f(x)
end
