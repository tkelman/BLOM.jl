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
    objcoefs::SparseMatrixCSC{Float64,Int} # actually a sparse vector
    constrcoefs::SparseMatrixCSC{Float64,Int}
    exponents::SparseMatrixCSC{Float64,Int}
    Model() = new(0, Float64[], Float64[], Float64[], Symbol[],
        spzeros(0,1), spzeros(0,0), spzeros(0,0))
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

function getValue(xi::Variable)
    return xi.model.x[xi.idx]
end

function setValue!(xi::Variable, v::Real)
    xi.model.x[xi.idx] = v
end

type GeneralExpression # linear combination of terms of the form ∏ᵢ x[i]^p[i]
    # this type may contain special functions and auxiliary equality constraints
    model::Model
    coefs::Vector{Float64}
    exponents::SparseMatrixCSC{Float64,Int}
    specialfcn::Bool
    auxK::SparseMatrixCSC{Float64,Int} # auxiliary constraint coefficient matrix
    auxPt::SparseMatrixCSC{Float64,Int} # auxiliary constraint exponent matrix
end

function Base.copy(ex::GeneralExpression)
    return GeneralExpression(ex.model, copy(ex.coefs), copy(ex.exponents),
        ex.specialfcn, copy(ex.auxK), copy(ex.auxPt))
end

function Base.convert(::Type{GeneralExpression}, x::Variable)
    model = x.model
    numvars = model.numvars
    return GeneralExpression(model, [1.0], sparsevec(x.idx, 1.0, numvars),
        false, spzeros(0, 0), spzeros(numvars, 0))
end

Base.promote_type(::Type{Variable}, ::Type{Variable}) = GeneralExpression
Base.promote_rule(::Type{Variable}, ::Type{GeneralExpression}) = GeneralExpression

include("functioncodes.jl")
include("operators.jl")

function getValue(ex::GeneralExpression)
    x = ex.model.x
    coefs = ex.coefs
    exponents = ex.exponents
    rowvals = exponents.rowval
    nzvals = exponents.nzval
    result = 0.0
    for col = 1:length(coefs)
        prodval = 1.0
        (imin, imax) = nzrange(SparseColumnView(exponents, col))
        for i = imin:imax
            prodval *= evalfunction(x[rowvals[i]], nzvals[i])
        end
        result += coefs[col] * prodval
    end
    return result
end

end # module
