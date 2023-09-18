using JSON
using Logging
using LoggingExtras
using Dates
using AdaProx

const date_format = "yyyy-mm-dd HH:MM:SS"

timestamped(logger) = TransformerLogger(logger) do log
    merge(log, (; message = "[$(Dates.format(now(), date_format))] $(log.message)"))
end

function is_logstep(base, it)
    scale = floor(Int, log(base, it))
    step = base ^ scale
    return mod(it, step) == 0
end

function get_logger(path, keys=nothing, base=10)
    format_logger = if keys === nothing
        FormatLogger(path) do io, args
            println(io, JSON.json(args.kwargs))
        end
    else
        FormatLogger(path) do io, args
            println(io, JSON.json(args.kwargs[keys]))
        end
    end
    TeeLogger(
        current_logger(),
        EarlyFilteredLogger(
            args -> args.level == AdaProx.Record,
            TeeLogger(
                format_logger,
                ActiveFilteredLogger(
                    args -> is_logstep(base, args.kwargs[:it]),
                    timestamped(ConsoleLogger(AdaProx.Record))
                )
            )
        ),
    )
end

_duration(df, key::Symbol) = maximum(df[!, key])

_duration(df, fun::Function) = maximum(fun(df))

function find_best(gb, names, objective_key, objective_target, duration_key)
    best_name, rest_names = Iterators.peel(names)
    best_duration = -1
    best_val = gb[best_name][!, objective_key][end]
    if best_val <= objective_target
        best_duration = _duration(gb[best_name], duration_key)
    end
    for name in rest_names
        duration = _duration(gb[name], duration_key)
        val = gb[name][!, objective_key][end]
        if val <= objective_target && (duration < best_duration || best_duration < 0)
            best_name = name
            best_duration = duration
        elseif best_duration < 0 && val < best_val
            best_name = name
            best_val = val
        end
    end
    return best_name
end
