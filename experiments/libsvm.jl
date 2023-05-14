using SparseArrays

function load_libsvm_dataset(file_path, ::Type{T} = Float64; labels=nothing) where {T}
    y = T[]

    rows = Int[]
    cols = Int[]
    vals = T[]

    if labels === nothing
        @info "Loading dataset $file_path as $T"
    else
        @info "Loading dataset $file_path as $T with labels $labels"

        @assert length(labels) == 2
        @assert labels[1] != labels[2]
        if labels[1] > labels[2]
            # NOTE do we really matter?
            @warn "Labels provided are not in ascending order"
        end
    end

    for (row, line) in enumerate(readlines(file_path))
        tokens = split(strip(line), " ")
        push!(y, parse(T, tokens[1]))
        for token in tokens[2:end]
            col_str, val_str = split(token, ":")
            push!(rows, row)
            push!(cols, parse(Int, col_str))
            push!(vals, parse(T, val_str))
        end
    end

    X = sparse(rows, cols, vals)

    @info "Number of data points: $(size(X, 1))"
    @info "Number of features: $(size(X, 2))"
    nnz_features = count(!iszero, X)
    @info "Density: $(nnz_features / prod(size(X)))"

    if labels !== nothing
        @assert length(unique(y)) == 2

        y0, y1 = minimum(y), maximum(y)
        l0, l1 = labels

        @info "Dataset has label values $y0 and $y1"

        if !(y0 in labels && y1 in labels)
            @info "Turning label values to $l0 and $l1"
            y[y.==y0] .= l0
            y[y.==y1] .= l1
        else
            @info "No label conversion needed"
        end

        @assert all(in(v, labels) for v in y)
    end

    return X, y
end
