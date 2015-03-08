module BLOM
using Calculus, Compat
import Base: copy, convert, promote_type, promote_rule
export getValue, setValue!

include("sparseutils.jl")

type Model
    numvars::Int
    x::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    vartypes::Vector{Symbol}
    objcoefs::SparseList{Float64,Int}
    constrcoefs::SparseMatrixASC{Float64,Int}
    exponents::SparseMatrixASC{Float64,Int}
    Model() = new(0, Float64[], Float64[], Float64[], Symbol[],
        slzeros(), asczeros(0,0), asczeros(0,0))
end
# For now, only do one type of Model with equality constraints and variable
# bounds. Later, look into a separate type of Model which tries harder to
# distribute multiplications instead of introducing auxiliary equality
# constraints, and allows inequality bounds on constraints too (for geometric
# programming, which should be doable as a special case here).

type Variable # should probably be immutable?
    model::Model
    idx::Int
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

type GeneralExpression # linear combination of terms of the form ∏ᵢ x[i]^p[i]
    # this type may contain special functions and auxiliary equality constraints
    model::Model
    coefs::Vector{Float64}
    exponents::SparseMatrixASC{Float64,Int}
    specialfcn::Bool
    auxK::SparseMatrixASC{Float64,Int} # auxiliary constraint coefficient matrix
    auxPt::SparseMatrixASC{Float64,Int} # auxiliary constraint exponent matrix
end
function GeneralExpression(model::Model, coefs::Vector{Float64},
        exponents::SparseMatrixASC{Float64,Int}, specialfcn::Bool)
    return GeneralExpression(model, coefs, exponents, specialfcn,
        asczeros(0, 0), asczeros(model.numvars, 0))
end

copy(ex::GeneralExpression) = GeneralExpression(ex.model, copy(ex.coefs),
    copy(ex.exponents), ex.specialfcn, copy(ex.auxK), copy(ex.auxPt))

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

end # module
