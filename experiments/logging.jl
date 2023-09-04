using JSON
using Logging
using LoggingExtras
using AdaProx

function get_logger(path, keys=nothing)
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
        EarlyFilteredLogger(
            args -> args.level != AdaProx.Record,
            ConsoleLogger()
        ),
        EarlyFilteredLogger(
            args -> args.level == AdaProx.Record,
            format_logger
        ),
    )
end
