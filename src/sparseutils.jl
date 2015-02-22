function sparsevec(idx::Int, val::Float64, numvars::Int)
    # construct a 1-column sparse matrix from a single nonzero
    return SparseMatrixCSC{Float64,Int}(numvars, 1, [1,2], [idx], [val])
end

immutable SparseColumnView
    A::SparseMatrixCSC{Float64,Int}
    colidx::Int
end

function nzrange(S::SparseColumnView)
    colptr = S.A.colptr
    col = S.colidx
    return (colptr[col], colptr[col+1]-1)
end

function lexcmp(col1::SparseColumnView, col2::SparseColumnView)
    # Return -1 if col1 has a nonzero in an earlier row than col2, or 1 if
    # col2 has a nonzero in an earlier row than col1. If both columns have
    # nonzeros in the same rows, return the first nonzero result of lexcmp
    # on the nonzero values, or 0 if all nonzero values are equal.
    (i1, i1max) = nzrange(col1)
    (i2, i2max) = nzrange(col2)
    if i1 > i1max # col1 is all zeros
        if i2 > i2max # col2 is also all zeros
            return 0
        else # col2 has at least 1 nonzero
            return 1
        end
    elseif i2 > i2max # col1 has at least 1 nonzero, col2 is all zeros
        return -1
    end
    rowvals1 = col1.A.rowval
    rowvals2 = col2.A.rowval
    nzvals1 = col1.A.nzval
    nzvals2 = col2.A.nzval
    while true
        r1 = rowvals1[i1]
        r2 = rowvals2[i2]
        if r1 == r2
            nz1 = nzvals1[i1]
            nz2 = nzvals2[i2]
            if nz1 < nz2
                return -1
            elseif nz1 > nz2
                return 1
            elseif i1 > i1max # col1 has no more nonzeros
                if i2 > i2max # col2 also has no more nonzeros
                    return 0
                else # col2 has at least 1 more nonzero
                    return 1
                end
            elseif i2 > i2max # col1 has at least 1 more nonzero, col2 doesn't
                return -1
            end
            i1 += 1
            i2 += 1
        elseif r1 < r2
            return -1
        else # r1 > r2
            return 1
        end
    end
end

function appendcolumn!(A::SparseMatrixCSC{Float64,Int}, col::SparseColumnView,
        rowoffset::Integer = 0)
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
        allow_inplace::Bool)
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
            coefsout[Pt1_n] += coefs2[1]
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
            push!(coefsout, coefs2[Pt2_c])
            Pt2_c += 1
            Pt2_cv = SparseColumnView(Pt2, Pt2_c)
        else
            appendcolumn!(Pto, Pt1_cv)
            push!(coefsout, coefs1[Pt1_c] + coefs2[Pt2_c])
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
        push!(coefsout, coefs2[c2])
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
            deleteat!(Ko.colptr, K1_n)
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

