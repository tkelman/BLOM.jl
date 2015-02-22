module BLOM
using Calculus, Compat

functioncodes = Dict{Symbol, Float64}()
for (i, f) in enumerate(symbolic_derivatives_1arg())
    # code value chosen large enough that (1+eps)^x overflows in Float64
    # compatibility risk if the existing order for these moves around
    functioncodes[f[1]] = i * 1.0e20
end

type Model
    numvars::Int
    x::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    objcoefs::SparseMatrixCSC{Float64,Int} # actually a sparse vector
    constrcoefs::SparseMatrixCSC{Float64,Int}
    exponents::SparseMatrixCSC{Float64,Int}
    Model() = new(0, Float64[], Float64[], Float64[], spzeros(0,1),
        spzeros(0,0), spzeros(0,0))
end
# For now, only do one type of Model with equality constraints and variable
# bounds. Later, look into a separate type of Model which tries harder to
# distribute multiplications instead of introducing auxiliary equality
# constraints, and allows inequality bounds on constraints too (for geometric
# programming, which should be doable as a special case here).

type Variable
    model::Model
    idx::Int
end

function getValue(xi::Variable)
    return xi.model.x[xi.idx]
end

function setValue!(xi::Variable, v::Number)
    xi.model.x[xi.idx] = v
end

function newVariable(model::Model, lb = -Inf, ub = Inf, x0 = NaN)
    model.numvars += 1
    push!(model.x, x0)
    push!(model.lb, lb)
    push!(model.ub, ub)
    model.exponents.m += 1
    return Variable(model, model.numvars)
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

function Base.convert(::Type{GeneralExpression}, x::Variable)
    model = x.model
    numvars = model.numvars
    return GeneralExpression(model, [1.0], sparsevec(x.idx, 1.0, numvars),
        false, spzeros(0, 0), spzeros(numvars, 0))
end

include("sparseutils.jl")
include("operators.jl")


end # module
