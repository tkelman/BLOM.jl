import Base: nnz, nonzeros, getindex, copy, convert, size, showarray, lexcmp
if VERSION >= v"0.4.0-dev+1307"
    import Base: rowvals
else
    nzrange(S::SparseMatrixCSC, col::Integer) =
        S.colptr[col]:(S.colptr[col+1]-1)
end

# this is effectively a sparse vector type, but does not store its size
type SparseList{Tv,Ti<:Integer}
    idx::Vector{Ti}   # indices of nonzeros, assumed to be strictly sorted
    nzval::Vector{Tv} # structurally nonzero values
end

nnz(S::SparseList) = length(S.idx)
nonzeros(S::SparseList) = S.nzval
rowvals(S::SparseList) = S.idx
isempty(S::SparseList) = (nnz(S) == 0)

function getindex{Tv,Ti}(S::SparseList{Tv,Ti}, i::Integer)
    (i < 1) && throw(BoundsError())
    idx = S.idx
    k = searchsortedfirst(idx, i)
    if k > length(idx) || i != idx[k]
        return zero(Tv)
    else
        return S.nzval[k]
    end
end

copy(S::SparseList) = SparseList(copy(S.idx), copy(S.nzval))

convert{Tv,Ti,TvS,TiS}(::Type{SparseList{Tv,Ti}}, S::SparseList{TvS,TiS}) =
    (Tv == TvS && Ti == TiS) ? S : SparseList(convert(Vector{Ti}, S.idx),
        convert(Vector{Tv}, S.nzval))

# Base.spzeros isn't extensible to other output formats
slzeros(Tv::Type = Float64, Ti::Type = Int) =
    SparseList(Array(Ti, 0), Array(Tv, 0))

if VERSION < v"0.4.0-dev+2014"
    function Base.sizehint(S::SparseList, n::Integer)
        sizehint(S.idx, n)
        sizehint(S.nzval, n)
        return S
    end
else
    function Base.sizehint!(S::SparseList, n::Integer)
        sizehint!(S.idx, n)
        sizehint!(S.nzval, n)
        return S
    end
end

binaryheader = quote idxA = A.idx; idxB = B.idx; nzvalA = A.nzval;
    nzvalB = B.nzval; nnzA = length(idxA); nnzB = length(idxB) end

for op in (:+, :.+, :-, :.-, :.*)
    @eval begin
        function ($op){TvA,TiA,TvB,TiB}(A::SparseList{TvA,TiA},
                B::SparseList{TvB,TiB})
            Tv = promote_type(TvA, TvB)
            Ti = promote_type(TiA, TiB)
            A = convert(SparseList{Tv,Ti}, A)
            B = convert(SparseList{Tv,Ti}, B)
            return ($op)(A, B)
        end

        function ($op){Tv,Ti}(A::SparseList{Tv,Ti}, B::SparseList{Tv,Ti})
            $binaryheader
            nnzS = nnzA + nnzB # conservative estimate
            idxS = Array(Ti, nnzS)
            nzvalS = Array(Tv, nnzS)
            z = zero(Tv)
            iS = 1
            @inbounds begin
                iA = 1
                iB = 1
                if iA <= nnzA && iB <= nnzB
                    rA = idxA[iA]
                    rB = idxB[iB]
                    while true
                        if rA < rB
                            res = ($op)(nzvalA[iA], z)
                            if res != z
                                idxS[iS] = rA
                                nzvalS[iS] = res
                                iS += 1
                            end
                            iA += 1
                            if iA <= nnzA
                                rA = idxA[iA]
                            else
                                break
                            end
                        elseif rB < rA
                            res = ($op)(z, nzvalB[iB])
                            if res != z
                                idxS[iS] = rB
                                nzvalS[iS] = res
                                iS += 1
                            end
                            iB += 1
                            if iB <= nnzB
                                rB = idxB[iB]
                            else
                                break
                            end
                        else
                            res = ($op)(nzvalA[iA], nzvalB[iB])
                            if res != z
                                idxS[iS] = rA
                                nzvalS[iS] = res
                                iS += 1
                            end
                            iA += 1
                            iB += 1
                            if iA <= nnzA && iB <= nnzB
                                rA = idxA[iA]
                                rB = idxB[iB]
                            else
                                break
                            end
                        end
                    end
                end
                while iA <= nnzA
                    res = ($op)(nzvalA[iA], z)
                    if res != z
                        idxS[iS] = idxA[iA]
                        nzvalS[iS] = res
                        iS += 1
                    end
                    iA += 1
                end
                while iB <= nnzB
                    res = ($op)(z, nzvalB[iB])
                    if res != z
                        idxS[iS] = idxB[iB]
                        nzvalS[iS] = res
                        iS += 1
                    end
                    iB += 1
                end
            end
            deleteat!(idxS, iS : nnzS)
            deleteat!(nzvalS, iS : nnzS)
            return SparseList(idxS, nzvalS)
        end
    end
end

# array of sparse columns matrix format
type SparseMatrixASC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                          # number of rows
    cols::Vector{SparseList{Tv,Ti}} # array of columns
end
SparseMatrixASC{Tv,Ti<:Integer}(m::Integer, cols::Vector{SparseList{Tv,Ti}}) =
    SparseMatrixASC(int(m), cols)

size(S::SparseMatrixASC) = (S.m, length(S.cols))
nnz(S::SparseMatrixASC) = mapreduce(nnz, +, 0, S.cols) # this is O(n), not O(1)

function getindex(S::SparseMatrixASC, i::Integer, j::Integer)
    (m, n) = size(S)
    (i < 1 || i > m) && throw(BoundsError())
    (j < 1 || j > n) && throw(BoundsError())
    return S.cols[j][i]
end

copy(S::SparseMatrixASC) = SparseMatrixASC(S.m, [copy(col) for col in S.cols])
shallowcopy(S::SparseMatrixASC) = SparseMatrixASC(S.m, [col for col in S.cols])

function convert{Tv,Ti,TvS,TiS}(::Type{SparseMatrixASC{Tv,Ti}},
        S::SparseMatrixASC{TvS,TiS})
    if Tv == TvS && Ti == TiS
        return S
    else
        return SparseMatrixASC(S.m, convert(Vector{SparseList{Tv,Ti}}, S.cols))
    end
end

convert{Tv,Ti}(::Type{SparseMatrixCSC{Tv,Ti}}, S::SparseMatrixASC{Tv,Ti}) =
    SparseMatrixCSC(S.m, size(S,2), cumsum(unshift!(map(nnz,S.cols), one(Ti))),
        vcat(map(rowvals, S.cols)...), vcat(map(nonzeros, S.cols)...))

convert{Tv,Ti}(::Type{SparseMatrixASC{Tv,Ti}}, S::SparseMatrixCSC{Tv,Ti}) =
    SparseMatrixASC(S.m, [SparseList(S.rowval[nzrange(S, col)],
        S.nzval[nzrange(S, col)]) for col=1:S.n])

# Base.spzeros isn't extensible to other output formats
asczeros(m::Integer, n::Integer) = asczeros(Float64, m, n)
asczeros(Tv::Type, m::Integer, n::Integer) =
    SparseMatrixASC(m, [slzeros(Tv, Int) for col=1:n])
asczeros(Tv::Type, Ti::Type, m::Integer, n::Integer) =
    SparseMatrixASC(m, [slzeros(Tv, Ti) for col=1:n])

# construct a 1-column SparseMatrixASC from a single nonzero
sparsevec{Tv,Ti<:Integer}(idx::Ti, val::Tv, numvars::Integer) =
    SparseMatrixASC(numvars, [SparseList([idx], [val])])

for op in (:+, :.+, :-, :.-, :.*)
    @eval begin
        function ($op){TvA,TiA,TvB,TiB}(A::SparseMatrixASC{TvA,TiA},
                B::SparseMatrixASC{TvB,TiB})
            Tv = promote_type(TvA, TvB)
            Ti = promote_type(TiA, TiB)
            A = convert(SparseMatrixASC{Tv,Ti}, A)
            B = convert(SparseMatrixASC{Tv,Ti}, B)
            return ($op)(A, B)
        end

        function ($op){Tv,Ti}(A::SparseMatrixASC{Tv,Ti},
                B::SparseMatrixASC{Tv,Ti})
            (m, n) = size(A)
            if (m, n) != size(B)
                throw(DimensionMismatch(""))
            end
            Acols = A.cols
            Bcols = B.cols
            return SparseMatrixASC(m, [($op)(Acols[col], Bcols[col]) for col=1:n])
        end
    end
end

function showarray(io::IO, S::SparseMatrixASC;
                   header::Bool=true, limit::Bool=Base._limit_output,
                   rows = Base.tty_size()[1], repr=false)
    (m, n) = size(S)
    if header
        print(io, m, "x", n, " sparse matrix with ", nnz(S), " ",
            eltype(S), " entries:")
    end

    if limit
        half_screen_rows = div(rows - 8, 2)
    else
        half_screen_rows = typemax(Int)
    end
    pad = ndigits(max(m, n))
    k = 0
    sep = "\n\t"
    for col = 1:n
        curcol = S.cols[col]
        idx = curcol.idx
        nzval = curcol.nzval
        for i = 1:length(idx)
            if k < half_screen_rows || k > nnz(S)-half_screen_rows
                print(io, sep, '[', rpad(idx[i], pad), ", ",
                    lpad(col, pad), "]  =  ")
                showcompact(io, nzval[i])
            elseif k == half_screen_rows
                print(io, sep, '\u22ee')
            end
            k += 1
        end
    end
end

# BLOM-specific definitions
@eval function lexcmp(A::SparseList, B::SparseList)
    # Return -1 if A has a nonzero in an earlier index than B, or 1 if
    # B has a nonzero in an earlier index than A. If both columns have
    # nonzeros in the same rows, return the first nonzero result of lexcmp
    # on the nonzero values, or 0 if all nonzero values are equal.
    $binaryheader
    iA = 1
    iB = 1
    while true
        if iA > nnzA # rest of A is all zeros
            if iB > nnzB # rest of B is also all zeros
                return 0
            else # B has at least 1 more nonzero
                return 1
            end
        elseif iB > nnzB # rest of B is all zeros
            return -1 # and A has at least 1 more nonzero
        end
        rA = idxA[iA]
        rB = idxB[iB]
        if rA == rB
            nz1 = nzvalA[iA]
            nz2 = nzvalB[iB]
            if nz1 < nz2
                return -1
            elseif nz1 > nz2
                return 1
            elseif iA > nnzA # A has no more nonzeros
                if iB > nnzB # B also has no more nonzeros
                    return 0
                else # B has at least 1 more nonzero
                    return 1
                end
            elseif iB > nnzB # A has at least 1 more nonzero, B doesn't
                return -1
            end
            iA += 1
            iB += 1
        elseif rA < rB
            return -1
        else # rA > rB
            return 1
        end
    end
end

@eval function vcat_offset{Tv,Ti}(A::SparseList{Tv,Ti}, B::SparseList{Tv,Ti},
        offset::Integer)
    $binaryheader
    if nnzA > 0 && nnzB > 0 && idxB[1] + offset <= idxA[nnzA]
        error("overlapping vcat of SparseLists not implemented")
    end
    nnzS = nnzA + nnzB
    idxS = Array(Ti, nnzS)
    nzvalS = Array(Tv, nnzS)
    @inbounds for i = 1:nnzA
        idxS[i] = idxA[i]
        nzvalS[i] = nzvalA[i]
    end
    @inbounds for i = 1:nnzB
        idxS[i + nnzA] = idxB[i] + offset
        nzvalS[i + nnzA] = nzvalB[i]
    end
    return SparseList(idxS, nzvalS)
end

@eval function append_offset!{Tv,Ti}(A::SparseList{Tv,Ti},
        B::SparseList{Tv,Ti}, offset::Integer)
    $binaryheader
    if nnzA > 0 && nnzB > 0 && idxB[1] + offset <= idxA[nnzA]
        error("overlapping append! of SparseLists not implemented")
    end
    nnzS = nnzA + nnzB
    sizehint!(idxA, nnzS)
    sizehint!(nzvalA, nnzS)
    @inbounds for i = 1:nnzB
        push!(idxA, idxB[i] + offset)
        push!(nzvalA, nzvalB[i])
    end
    return A
end

function offset_rows!(A::SparseMatrixASC, offset::Integer)
    if offset == 0
        return A
    end
    A.m += offset
    for col in A.cols
        idx = col.idx
        for i = 1:length(idx)
            idx[i] += offset
        end
    end
    return A
end

# add 2 scalar expressions
function add_expressions(coefs1::Vector{Float64},
        Pt1::SparseMatrixASC{Float64,Int32},
        coefs2::Vector{Float64},
        Pt2::SparseMatrixASC{Float64,Int32},
        allow_inplace::Bool,
        mult::Float64 = 1.0)
    (Pt1_m, Pt1_n) = size(Pt1)
    (Pt2_m, Pt2_n) = size(Pt2)
    Pt1_cols = Pt1.cols
    Pt2_cols = Pt2.cols
    Pt2_c = 1
    if Pt2_c > Pt2_n
        if allow_inplace
            return (coefs1, Pt1)
        else
            return (copy(coefs1), copy(Pt1))
        end
    elseif Pt1_n == 0
        if allow_inplace
            return (scale!(coefs2, mult), Pt2)
        else
            return (mult .* coefs2, copy(Pt2))
        end
    end
    Pt2_col = Pt2_cols[Pt2_c]
    lc = lexcmp(Pt1_cols[Pt1_n], Pt2_col)
    if lc <= 0
        if allow_inplace
            coefsout = coefs1
            Pto = Pt1
        else
            coefsout = copy(coefs1)
            Pto = copy(Pt1)
        end
        Pt1_c = Pt1_n + 1
        if lc == 0
            newcoef = coefsout[Pt1_n] + mult * coefs2[1]
            if newcoef == 0.0
                # drop the already-copied last column of Pt1
                pop!(coefsout)
                pop!(Pto.cols)
            else
                coefsout[Pt1_n] = newcoef
            end
            Pt2_c = 2
            (Pt2_c <= Pt2_n) && (Pt2_col = Pt2_cols[Pt2_c])
        end
    else
        # TODO: check other direction, if last column of Pt2
        # sorts before first column of Pt1
        coefsout = Float64[]
        Pto = asczeros(Float64, Int32, Pt1_m, 0)
        Pt1_c = 1
    end
    Pto_cols = Pto.cols
    sizehint!(coefsout, Pt1_n + Pt2_n)
    sizehint!(Pto_cols, Pt1_n + Pt2_n)

    if (Pt1_c <= Pt1_n) && (Pt2_c <= Pt2_n)
        Pt1_col = Pt1_cols[Pt1_c]
        while true
            lc = lexcmp(Pt1_col, Pt2_col)
            if lc == -1
                push!(coefsout, coefs1[Pt1_c])
                push!(Pto_cols, Pt1_col)
                Pt1_c += 1
                if Pt1_c > Pt1_n
                    break
                else
                    Pt1_col = Pt1_cols[Pt1_c]
                end
            elseif lc == 1
                push!(coefsout, mult * coefs2[Pt2_c])
                push!(Pto_cols, Pt2_col)
                Pt2_c += 1
                if Pt2_c > Pt2_n
                    break
                else
                    Pt2_col = Pt2_cols[Pt2_c]
                end
            else
                newcoef = coefs1[Pt1_c] + mult * coefs2[Pt2_c]
                if newcoef != 0.0
                    push!(coefsout, newcoef)
                    push!(Pto_cols, Pt1_col)
                end
                Pt1_c += 1
                Pt2_c += 1
                if Pt1_c > Pt1_n || Pt2_c > Pt2_n
                    break
                else
                    Pt1_col = Pt1_cols[Pt1_c]
                    Pt2_col = Pt2_cols[Pt2_c]
                end
            end
        end
    end
    for c1 = Pt1_c : Pt1_n
        push!(coefsout, coefs1[c1])
        push!(Pto_cols, Pt1_cols[c1])
    end
    for c2 = Pt2_c : Pt2_n
        push!(coefsout, mult * coefs2[c2])
        push!(Pto_cols, Pt2_cols[c2])
    end
    return (coefsout, Pto)
end

# concatenate 2 vector expressions
function concat_expressions(K1::SparseMatrixASC{Float64,Int32},
        Pt1::SparseMatrixASC{Float64,Int32},
        K2::SparseMatrixASC{Float64,Int32},
        Pt2::SparseMatrixASC{Float64,Int32},
        allow_inplace::Bool)
    (K1_m, K1_n) = size(K1)
    (K2_m, K2_n) = size(K2)
    K1_cols = K1.cols
    K2_cols = K2.cols
    (Pt1_m, Pt1_n) = size(Pt1)
    (Pt2_m, Pt2_n) = size(Pt2)
    Pt1_cols = Pt1.cols
    Pt2_cols = Pt2.cols
    @assert K1_n == Pt1_n
    @assert K2_n == Pt2_n
    Pt2_c = 1
    if Pt2_c > Pt2_n
        if allow_inplace
            K1.m += K2_m
            return (K1, Pt1)
        else
            Ko = copy(K1)
            Ko.m += K2_m
            return (Ko, copy(Pt1))
        end
    elseif Pt1_n == 0
        if allow_inplace
            return (offset_rows!(K2, K1_m), Pt2)
        else
            return (offset_rows!(copy(K2), K1_m), copy(Pt2))
        end
    end
    Pt2_col = Pt2_cols[Pt2_c]
    lc = lexcmp(Pt1_cols[Pt1_n], Pt2_col)
    if lc <= 0
        if allow_inplace
            Ko = K1
            Pto = Pt1
        else
            Ko = copy(K1)
            Pto = copy(Pt1)
        end
        Pt1_c = Pt1_n + 1
        if lc == 0
            # append first column of K2 to end of Ko
            append_offset!(Ko.cols[Pt1_n], K2_cols[Pt2_c], K1_m)
            Pt2_c = 2
            (Pt2_c <= Pt2_n) && (Pt2_col = Pt2_cols[Pt2_c])
        end
        Ko.m += K2_m
    else
        # TODO: check other direction, if last column of Pt2
        # sorts before first column of Pt1
        Ko = asczeros(Float64, Int32, K1_m + K2_m, 0)
        Pto = asczeros(Float64, Int32, Pt1_m, 0)
        Pt1_c = 1
    end
    Ko_cols = Ko.cols
    Pto_cols = Pto.cols
    sizehint!(Ko_cols, K1_n + K2_n)
    sizehint!(Pto_cols, Pt1_n + Pt2_n)

    if (Pt1_c <= Pt1_n) && (Pt2_c <= Pt2_n)
        Pt1_col = Pt1_cols[Pt1_c]
        while true
            lc = lexcmp(Pt1_col, Pt2_col)
            if lc == -1
                push!(Ko_cols, K1_cols[Pt1_c])
                push!(Pto_cols, Pt1_col)
                Pt1_c += 1
                if Pt1_c > Pt1_n
                    break
                else
                    Pt1_col = Pt1_cols[Pt1_c]
                end
            elseif lc == 1
                Ko_col = K2_cols[Pt2_c]
                if allow_inplace
                    Ko_col.idx[:] += K1_m
                else
                    Ko_col = SparseList(Ko_col.idx + K1_m, Ko_col.nzval)
                end
                push!(Ko_cols, Ko_col)
                push!(Pto_cols, Pt2_col)
                Pt2_c += 1
                if Pt2_c > Pt2_n
                    break
                else
                    Pt2_col = Pt2_cols[Pt2_c]
                end
            else
                K1_col = K1_cols[Pt1_c]
                K2_col = K2_cols[Pt2_c]
                if allow_inplace
                    Ko_col = append_offset!(K1_col, K2_col, K1_m)
                else
                    Ko_col = vcat_offset(K1_col, K2_col, K1_m)
                end
                push!(Ko_cols, Ko_col)
                push!(Pto_cols, Pt1_col)
                Pt1_c += 1
                Pt2_c += 1
                if Pt1_c > Pt1_n || Pt2_c > Pt2_n
                    break
                else
                    Pt1_col = Pt1_cols[Pt1_c]
                    Pt2_col = Pt2_cols[Pt2_c]
                end
            end
        end
    end
    for c1 = Pt1_c : Pt1_n
        push!(Ko_cols, K1_cols[c1])
        push!(Pto_cols, Pt1_cols[c1])
    end
    for c2 = Pt2_c : Pt2_n
        Ko_col = K2_cols[c2]
        if allow_inplace
            Ko_col.idx[:] += K1_m
        else
            Ko_col = SparseList(Ko_col.idx + K1_m, Ko_col.nzval)
        end
        push!(Ko_cols, Ko_col)
        push!(Pto_cols, Pt2_cols[c2])
    end
    return (Ko, Pto)
end
