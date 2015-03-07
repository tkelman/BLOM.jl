import Base: nnz, nonzeros, convert, size, showarray, lexcmp
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
    n::Int                          # number of columns
    cols::Vector{SparseList{Tv,Ti}} # array of columns
end

size(S::SparseMatrixASC) = (S.m, S.n)
nnz(S::SparseMatrixASC) = mapreduce(nnz, +, 0, S.cols) # this is O(n), not O(1)

function convert{Tv,Ti,TvS,TiS}(::Type{SparseMatrixASC{Tv,Ti}},
        S::SparseMatrixASC{TvS,TiS})
    if Tv == TvS && Ti == TiS
        return S
    else
        return SparseMatrixASC(S.m, S.n,
            convert(Vector{SparseList{Tv,Ti}}, S.cols))
    end
end

convert{Tv,Ti}(::Type{SparseMatrixCSC{Tv,Ti}}, S::SparseMatrixASC{Tv,Ti}) =
    SparseMatrixCSC(S.m, S.n, cumsum(unshift!(map(nnz, S.cols), one(Ti))),
        vcat(map(rowvals, S.cols)...), vcat(map(nonzeros, S.cols)...))

convert{Tv,Ti}(::Type{SparseMatrixASC{Tv,Ti}}, S::SparseMatrixCSC{Tv,Ti}) =
    SparseMatrixASC(S.m, S.n, [SparseList(S.rowval[nzrange(S, col)],
        S.nzval[nzrange(S, col)]) for col=1:S.n])

# Base.spzeros isn't extensible to other output formats
asczeros(m::Integer, n::Integer) = asczeros(Float64, m, n)
asczeros(Tv::Type, m::Integer, n::Integer) =
    SparseMatrixASC(m, n, [slzeros(Tv, Int) for col=1:n])
asczeros(Tv::Type, Ti::Type, m::Integer, n::Integer) =
    SparseMatrixASC(m, n, [slzeros(Tv, Ti) for col=1:n])

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
            return SparseMatrixASC(m, n,
                [($op)(Acols[col], Bcols[col]) for col=1:n])
        end
    end
end

function showarray(io::IO, S::SparseMatrixASC;
                   header::Bool=true, limit::Bool=Base._limit_output,
                   rows = Base.tty_size()[1], repr=false)
    if header
        print(io, S.m, "x", S.n, " sparse matrix with ", nnz(S), " ",
            eltype(S), " entries:")
    end

    if limit
        half_screen_rows = div(rows - 8, 2)
    else
        half_screen_rows = typemax(Int)
    end
    pad = ndigits(max(S.m,S.n))
    k = 0
    sep = "\n\t"
    for col = 1:S.n
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
    nnzS = nnzA + nnzB
    idxS = Array(Ti, nnzS)
    nzvalS = Array(Tv, nnzS)
    if nnzA > 0 && nnzB > 0 && idxB[1] + offset <= idxA[nnzA]
        error("overlapping vcat of SparseLists not implemented")
    end
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

#=
function appendcolumn!(A::SparseMatrixCSC{Float64,Int},
        col::SparseColumnView, rowoffset::Integer = 0)
    out_colptr = A.colptr
    out_rowval = A.rowval
    out_nzval = A.nzval
    in_rowval = col.A.rowval
    in_nzval = col.A.nzval
    (imin, imax) = nzrange(col)
    new_nnz = out_colptr[end] + imax - imin
    sizehint!(out_rowval, new_nnz)
    sizehint!(out_nzval, new_nnz)
    for i = imin : imax
        push!(out_rowval, in_rowval[i] + rowoffset)
        push!(out_nzval, in_nzval[i])
    end
    push!(out_colptr, new_nnz + 1)
    A.n += 1
    return A
end

function appendtwocolumns!(A::SparseMatrixCSC{Float64,Int},
        col1::SparseColumnView, col2::SparseColumnView)
    # stack col1 and col2 into same new column at end of A
    out_colptr = A.colptr
    out_rowval = A.rowval
    out_nzval = A.nzval
    in1_rowval = col1.A.rowval
    in1_nzval = col1.A.nzval
    (imin1, imax1) = nzrange(col1)
    rowoffset = col1.A.m
    in2_rowval = col2.A.rowval
    in2_nzval = col2.A.nzval
    (imin2, imax2) = nzrange(col2)
    new_nnz = out_colptr[end] + imax1 - imin1 + imax2 - imin2 + 1
    sizehint!(out_rowval, new_nnz)
    sizehint!(out_nzval, new_nnz)
    for i = imin1 : imax1
        push!(out_rowval, in1_rowval[i])
        push!(out_nzval, in1_nzval[i])
    end
    for i = imin2 : imax2
        push!(out_rowval, in2_rowval[i] + rowoffset)
        push!(out_nzval, in2_nzval[i])
    end
    push!(out_colptr, new_nnz + 1)
    A.n += 1
    return A
end

# add 2 scalar expressions
function add_expressions(coefs1::Vector{Float64},
        Pt1::SparseMatrixCSC{Float64,Int},
        coefs2::Vector{Float64},
        Pt2::SparseMatrixCSC{Float64,Int},
        allow_inplace::Bool,
        mult::Float64 = 1.0)
    (Pt1_m, Pt1_n) = size(Pt1)
    (Pt2_m, Pt2_n) = size(Pt2)
    Pt1_nnz = nnz(Pt1)
    Pt2_nnz = nnz(Pt2)
    Pt2_c = 1
    Pt2_cv = SparseColumnView(Pt2, Pt2_c)
    lc = lexcmp(SparseColumnView(Pt1, Pt1_n), Pt2_cv)
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
                Pto.n -= 1
                (imin, imax) = nzrange(SparseColumnView(Pt1, Pt1_n))
                deleteat!(Pto.rowval, imin : imax)
                deleteat!(Pto.nzval, imin : imax)
                pop!(Pto.colptr)
            else
                coefsout[Pt1_n] = newcoef
            end
            Pt2_c = 2
            Pt2_cv = SparseColumnView(Pt2, Pt2_c)
        end
    else
        # TODO: check other direction, if last column of Pt2 sorts before
        # first column of Pt1
        coefsout = Float64[]
        Pto = spzeros(Pt1_m, 0)
        Pt1_c = 1
    end
    sizehint!(coefsout, Pt1_n + Pt2_n)
    sizehint!(Pto.colptr, Pt1_n + Pt2_n + 1)
    sizehint!(Pto.rowval, Pt1_nnz + Pt2_nnz)
    sizehint!(Pto.nzval, Pt1_nnz + Pt2_nnz)

    Pt1_cv = SparseColumnView(Pt1, Pt1_c)
    while (Pt1_c <= Pt1_n) && (Pt2_c <= Pt2_n)
        lc = lexcmp(Pt1_cv, Pt2_cv)
        if lc == -1
            appendcolumn!(Pto, Pt1_cv)
            push!(coefsout, coefs1[Pt1_c])
            Pt1_c += 1
            Pt1_cv = SparseColumnView(Pt1, Pt1_c)
        elseif lc == 1
            appendcolumn!(Pto, Pt2_cv)
            push!(coefsout, mult * coefs2[Pt2_c])
            Pt2_c += 1
            Pt2_cv = SparseColumnView(Pt2, Pt2_c)
        else
            newcoef = coefs1[Pt1_c] + mult * coefs2[Pt2_c]
            if newcoef != 0
                appendcolumn!(Pto, Pt1_cv)
                push!(coefsout, newcoef)
            end
            Pt1_c += 1
            Pt2_c += 1
            Pt1_cv = SparseColumnView(Pt1, Pt1_c)
            Pt2_cv = SparseColumnView(Pt2, Pt2_c)
        end
    end
    for c1 = Pt1_c : Pt1_n
        appendcolumn!(Pto, SparseColumnView(Pt1, c1))
        push!(coefsout, coefs1[c1])
    end
    for c2 = Pt2_c : Pt2_n
        appendcolumn!(Pto, SparseColumnView(Pt2, c2))
        push!(coefsout, mult * coefs2[c2])
    end
    return (coefsout, Pto)
end

# concatenate 2 vector expressions
function concat_expressions(K1::SparseMatrixCSC{Float64,Int},
        Pt1::SparseMatrixCSC{Float64,Int},
        K2::SparseMatrixCSC{Float64,Int},
        Pt2::SparseMatrixCSC{Float64,Int},
        allow_inplace::Bool)
    (K1_m, K1_n) = size(K1)
    (K2_m, K2_n) = size(K2)
    K1_nnz = nnz(K1)
    K2_nnz = nnz(K2)
    (Pt1_m, Pt1_n) = size(Pt1)
    (Pt2_m, Pt2_n) = size(Pt2)
    Pt1_nnz = nnz(Pt1)
    Pt2_nnz = nnz(Pt2)
    @assert K1_n == Pt1_n
    @assert K2_n == Pt2_n
    Pt2_c = 1
    Pt2_cv = SparseColumnView(Pt2, Pt2_c)
    lc = lexcmp(SparseColumnView(Pt1, Pt1_n), Pt2_cv)
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
            appendcolumn!(Ko, SparseColumnView(K2, Pt2_c), K1_m)
            # append first column of K2 to end of Ko, but adjust colptr and n to
            # put it in the same column as the already-copied last column of K1
            K1_n > 0 && deleteat!(Ko.colptr, K1_n)
            Ko.n -= 1
            Pt2_c = 2
            Pt2_cv = SparseColumnView(Pt2, Pt2_c)
        end
        Ko.m += K2_m
    else
        # TODO: check other direction, if last column of Pt2 sorts before
        # first column of Pt1
        Ko = spzeros(K1_m + K2_m, 0)
        Pto = spzeros(Pt1_m, 0)
        Pt1_c = 1
    end
    sizehint!(Ko.colptr, K1_n + K2_n + 1)
    sizehint!(Ko.rowval, K1_nnz + K2_nnz)
    sizehint!(Ko.nzval, K1_nnz + K2_nnz)
    sizehint!(Pto.colptr, Pt1_n + Pt2_n + 1)
    sizehint!(Pto.rowval, Pt1_nnz + Pt2_nnz)
    sizehint!(Pto.nzval, Pt1_nnz + Pt2_nnz)

    Pt1_cv = SparseColumnView(Pt1, Pt1_c)
    while (Pt1_c <= Pt1_n) && (Pt2_c <= Pt2_n)
        lc = lexcmp(Pt1_cv, Pt2_cv)
        if lc == -1
            appendcolumn!(Pto, Pt1_cv)
            appendcolumn!(Ko, SparseColumnView(K1, Pt1_c))
            Pt1_c += 1
            Pt1_cv = SparseColumnView(Pt1, Pt1_c)
        elseif lc == 1
            appendcolumn!(Pto, Pt2_cv)
            appendcolumn!(Ko, SparseColumnView(K2, Pt2_c), K1_m)
            Pt2_c += 1
            Pt2_cv = SparseColumnView(Pt2, Pt2_c)
        else
            appendcolumn!(Pto, Pt1_cv)
            appendtwocolumns!(Ko, SparseColumnView(K1, Pt1_c),
                SparseColumnView(K2, Pt2_c))
            Pt1_c += 1
            Pt2_c += 1
            Pt1_cv = SparseColumnView(Pt1, Pt1_c)
            Pt2_cv = SparseColumnView(Pt2, Pt2_c)
        end
    end
    for c1 = Pt1_c : Pt1_n
        appendcolumn!(Pto, SparseColumnView(Pt1, c1))
        appendcolumn!(Ko, SparseColumnView(K1, c1))
    end
    for c2 = Pt2_c : Pt2_n
        appendcolumn!(Pto, SparseColumnView(Pt2, c2))
        appendcolumn!(Ko, SparseColumnView(K2, c2))
    end
    return (Ko, Pto)
end
=#
