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
