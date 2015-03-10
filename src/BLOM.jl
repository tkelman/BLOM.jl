module BLOM
using Calculus, Compat
import Base: copy, convert, promote_type, promote_rule, show
export getValue, setValue!

include("sparseutils.jl")

type Model
    numvars::Int32
    x::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    vartypes::Vector{Symbol}
    objsense::Symbol
    objcoefs::SparseList{Float64,Int32}
    constrcoefs::SparseMatrixASC{Float64,Int32}
    exponents::SparseMatrixASC{Float64,Int32}
    Model() = new(0, Float64[], Float64[], Float64[], Symbol[], :Min,
        slzeros(Float64, Int32), asczeros(Float64, Int32, 0, 0),
        asczeros(Float64, Int32, 0, 0))
end
# For now, only do one type of Model with equality constraints and variable
# bounds. Later, look into a separate type of Model which tries harder to
# distribute multiplications instead of introducing auxiliary equality
# constraints, and allows inequality bounds on constraints too (for geometric
# programming, which should be doable as a special case here).

immutable Variable
    model::Model
    idx::Int32
end

function Variable(model::Model; lb = -Inf, ub = Inf, start = NaN, vartype = :Cont)
    model.numvars += 1
    push!(model.x, start)
    push!(model.lb, lb)
    push!(model.ub, ub)
    push!(model.vartypes, vartype)
    model.exponents.m += 1
    return Variable(model, model.numvars)
end

getValue(xi::Variable) = xi.model.x[xi.idx]

function setValue!(xi::Variable, v::Real)
    xi.model.x[xi.idx] = v
end

function isconstrained(xi::Variable)
    model = xi.model
    idx = xi.idx
    return isfinite(model.lb[idx]) || isfinite(model.ub[idx]) ||
        model.vartypes[idx] != :Cont
end

type GeneralExpression # linear combination of terms of the form ∏ᵢ x[i]^p[i]
    # this type may contain special functions and auxiliary equality constraints
    model::Model
    coefs::Vector{Float64}
    exponents::SparseMatrixASC{Float64,Int32}
    specialfcn::Bool
    auxK::SparseMatrixASC{Float64,Int32} # auxiliary constraint coefficient matrix
    auxPt::SparseMatrixASC{Float64,Int32} # auxiliary constraint exponent matrix
end
function GeneralExpression(model::Model, coefs::Vector{Float64},
        exponents::SparseMatrixASC{Float64,Int32}, specialfcn::Bool)
    return GeneralExpression(model, coefs, exponents, specialfcn,
        asczeros(Float64,Int32,0,0), asczeros(Float64,Int32,model.numvars,0))
end

copy(ex::GeneralExpression) = GeneralExpression(ex.model, copy(ex.coefs),
    copy(ex.exponents), ex.specialfcn, copy(ex.auxK), copy(ex.auxPt))
shallowcopy(ex::GeneralExpression) = GeneralExpression(ex.model,
    copy(ex.coefs), shallowcopy(ex.exponents), ex.specialfcn,
    shallowcopy(ex.auxK), shallowcopy(ex.auxPt))

convert(::Type{GeneralExpression}, x::Variable) = GeneralExpression(x.model,
    [1.0], sparsevec(x.idx, 1.0, x.model.numvars), false)

promote_type(::Type{Variable}, ::Type{Variable}) = GeneralExpression
promote_rule(::Type{Variable}, ::Type{GeneralExpression}) = GeneralExpression

include("functioncodes.jl")
include("operators.jl")

function getValue(ex::GeneralExpression)
    coefs = ex.coefs
    exponents = ex.exponents
    cols = exponents.cols
    @assert length(coefs) == length(cols)
    x = ex.model.x
    result = 0.0
    @inbounds for col = 1:length(coefs)
        curcol = cols[col]
        rowvals = curcol.idx
        nzvals = curcol.nzval
        prodval = 1.0
        for i = 1:length(rowvals)
            prodval *= evalfunction(x[rowvals[i]], nzvals[i])
        end
        result += coefs[col] * prodval
    end
    return result
end

toExpr(x::Variable) = Expr(:ref, :x, x.idx)

function toExpr(sl::SparseList)
    idx = sl.idx
    nzval = sl.nzval
    @assert length(idx) == length(nzval)
    if length(idx) == 0
        return 1.0
    end
    out = Expr(:call, :*)
    args = out.args
    for i = 1:length(idx)
        var = Expr(:ref, :x, idx[i])
        p = nzval[i]
        if p == 1.0
            push!(args, var)
        elseif p < minfunctioncode
            push!(args, Expr(:call, :^, var, p))
        elseif haskey(specialfunctions, p)
            push!(args, Expr(:call, specialfunctions[p], var))
        else
            error("function not found for code $p")
        end
    end
    return length(idx) == 1 ? args[2] : out
end

function toExpr(coefs::Vector{Float64},
        exponents::SparseMatrixASC{Float64,Int32})
    cols = exponents.cols
    @assert length(coefs) == length(cols)
    if length(coefs) == 0
        return 0.0
    end
    out = Expr(:call, :+)
    args = out.args
    for i = 1:length(coefs)
        coefi = coefs[i]
        term = toExpr(cols[i])
        if coefi == 1.0
            push!(args, term)
        elseif term == 1.0
            push!(args, coefi)
        else
            push!(args, Expr(:call, :*, coefi, term))
        end
    end
    return length(coefs) == 1 ? args[2] : out
end

function toExpr(K::SparseMatrixASC{Float64,Int32},
        Pt::SparseMatrixASC{Float64,Int32})
    (K_m, K_n) = size(K)
    (Pt_m, Pt_n) = size(Pt)
    K_cols = K.cols
    Pt_cols = Pt.cols
    @assert K_n == Pt_n == length(K_cols) == length(Pt_cols)
    out = Any[Expr(:call, :+) for row = 1:K_m]
    for col = 1:K_n
        term = toExpr(Pt_cols[col])
        K_col = K_cols[col]
        rowvals = K_col.idx
        nzvals = K_col.nzval
        for i = 1:length(rowvals)
            coefi = nzvals[i]
            args = out[rowvals[i]].args
            if coefi == 1.0
                push!(args, term)
            elseif term == 1.0
                push!(args, coefi)
            else
                push!(args, Expr(:call, :*, coefi, term))
            end
        end
    end
    for row = 1:K_m
        args = out[row].args
        if length(args) == 1
            out[row] = 0.0
        elseif length(args) == 2
            out[row] = args[2]
        end
    end
    return out
end

show(io::IO, x::Variable) = print(io, toExpr(x))

function show(io::IO, ex::GeneralExpression)
    print(io, toExpr(ex.coefs, ex.exponents))
    constr = toExpr(ex.auxK, ex.auxPt)
    if length(constr) > 0
        print(io, "\nsubject to:")
        for con in constr
            print(io, "\n\t", con, " == 0")
        end
    end
end

function show(io::IO, model::Model)
    numvars = model.numvars
    objcoefs = model.objcoefs
    idx = objcoefs.idx
    exponents = model.exponents
    objexponents = SparseMatrixASC(numvars, length(idx), exponents.cols[idx])
    print(io, lowercase(string(model.objsense)), " ",
        toExpr(objcoefs.nzval, objexponents))
    constr = toExpr(model.constrcoefs, exponents)
    if length(constr) > 0 ||
            any(i -> isconstrained(Variable(model, i)), 1:numvars)
        print(io, "\nsubject to:")
    end
    for con in constr
        print(io, "\n\t", con, " == 0")
    end
    lb = model.lb
    ub = model.ub
    vartypes = model.vartypes
    for i = 1:numvars
        vari = Variable(model, i)
        isconstrained(vari) || continue
        print(io, "\n\t")
        lbi = lb[i]
        ubi = ub[i]
        vartypei = vartypes[i]
        isfinite(lbi) && print(io, lbi, " ≤ ")
        print(io, vari)
        isfinite(ubi) && print(io, " ≤ ", ubi)
        if vartypei != :Cont
            (isfinite(lbi) || isfinite(ubi)) && print(io, ", ", vari)
            if vartypei == :Int
                print(io, " Integer")
            elseif vartypei == :Bin
                print(io, " Binary")
            elseif vartypei == :SemiCont
                print(io, " SemiContinuous")
            elseif vartypei == :SemiInt
                print(io, " SemiInteger")
            elseif vartypei == :Fixed
                print(io, " Fixed")
            else
                error("unknown vartype $vartypei")
            end
        end
    end
end

end # module
