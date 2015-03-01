# in-place add! that mutates the first argument, for conversion of expressions
# where we don't have to worry about the same object being used elsewhere
function add!(ex1::GeneralExpression, ex2::GeneralExpression)
    ex1.model === ex2.model || error("expressions must be from same model")
    ex1.specialfcn |= ex2.specialfcn
    (ex1.coefs, ex1.exponents) = add_expressions(ex1.coefs, ex1.exponents,
        ex2.coefs, ex2.exponents, true)
    (ex1.auxK, ex1.auxPt) = concat_expressions(ex1.auxK, ex1.auxPt,
        ex2.auxK, ex2.auxPt, true)
    return ex1
end

# copying + for general operator overloading purposes
function +(ex1::GeneralExpression, ex2::GeneralExpression)
    model = ex1.model
    model === ex2.model || error("expressions must be from same model")
    (coefs, exponents) = add_expressions(ex1.coefs, ex1.exponents,
        ex2.coefs, ex2.exponents, false)
    (auxK, auxPt) = concat_expressions(ex1.auxK, ex1.auxPt,
        ex2.auxK, ex2.auxPt, false)
    return GeneralExpression(model, coefs, exponents,
        ex1.specialfcn || ex2.specialfcn, auxK, auxPt)
end

function minus!(ex1::GeneralExpression, ex2::GeneralExpression)
    ex1.model === ex2.model || error("expressions must be from same model")
    ex1.specialfcn |= ex2.specialfcn
    (ex1.coefs, ex1.exponents) = add_expressions(ex1.coefs, ex1.exponents,
        ex2.coefs, ex2.exponents, true, -1.0)
    (ex1.auxK, ex1.auxPt) = concat_expressions(ex1.auxK, ex1.auxPt,
        ex2.auxK, ex2.auxPt, true)
    return ex1
end

function -(ex1::GeneralExpression, ex2::GeneralExpression)
    model = ex1.model
    model === ex2.model || error("expressions must be from same model")
    (coefs, exponents) = add_expressions(ex1.coefs, ex1.exponents,
        ex2.coefs, ex2.exponents, false, -1.0)
    (auxK, auxPt) = concat_expressions(ex1.auxK, ex1.auxPt,
        ex2.auxK, ex2.auxPt, false)
    return GeneralExpression(model, coefs, exponents,
        ex1.specialfcn || ex2.specialfcn, auxK, auxPt)
end

function scale!(ex1::GeneralExpression, v2::Real)
    scale!(ex1.coefs, v2)
    return ex1
end
scale(ex1::GeneralExpression, v2::Real) = scale!(copy(ex1), v2)

function num2expr(v1::Real, model::Model)
    numvars = model.numvars
    return GeneralExpression(model, Float64[v1], spzeros(numvars, 1),
        false, spzeros(0, 0), spzeros(numvars, 0))
end

+(ex1::GeneralExpression) = copy(ex1)
-(ex1::GeneralExpression) = GeneralExpression(ex1.model, -ex1.coefs,
    copy(ex1.exponents), ex1.specialfcn, copy(ex1.auxK), copy(ex1.auxPt))
+(x1::Variable) = x1
-(x1::Variable) = scale!(convert(GeneralExpression, x1), -1.0)

function add!(ex1::GeneralExpression, v2::Real)
    if v2 == 0.0
        return ex1
    end
    coefs = ex1.coefs
    exponents = ex1.exponents
    colptr = exponents.colptr
    if length(colptr) == 0
        error("invalid expression")
    elseif length(coefs) == 0 || colptr[end-1] != colptr[end]
        push!(coefs, v2)
        push!(colptr, colptr[end])
        exponents.n += 1
    else
        newval = coefs[end] + v2
        if newval != 0.0
            coefs[end] = newval
        else
            pop!(coefs)
            pop!(colptr)
            exponents.n -= 1
        end
    end
    return ex1
end
add!(v1::Real, ex2::GeneralExpression) = add!(ex2, v1)
+(ex1::GeneralExpression, v2::Real) = add!(copy(ex1), v2)
+(v1::Real, ex2::GeneralExpression) = add!(copy(ex2), v1)
minus!(ex1::GeneralExpression, v2::Real) = add!(ex1, -v2)
minus!(v1::Real, ex2::GeneralExpression) = add!(scale!(ex2, -1.0), v1)
-(ex1::GeneralExpression, v2::Real) = add!(copy(ex1), -v2)
-(v1::Real, ex2::GeneralExpression) = add!(-ex2, v1)
mul!(ex1::GeneralExpression, v2::Real) = scale!(ex1, v2)
mul!(v1::Real, ex2::GeneralExpression) = scale!(ex2, v1)
*(ex1::GeneralExpression, v2::Real) = scale!(copy(ex1), v2)
*(v1::Real, ex2::GeneralExpression) = scale!(copy(ex2), v1)
rdiv!(ex1::GeneralExpression, v2::Real) = scale!(ex1, 1.0 / v2)
rdiv!(v1::Real, ex2::GeneralExpression) = rdiv!(num2expr(v1, ex2.model), ex2)
/(ex1::GeneralExpression, v2::Real) = scale!(copy(ex1), 1.0 / v2)
/(v1::Real, ex2::GeneralExpression) = rdiv!(num2expr(v1, ex2.model), copy(ex2))

add!(ex1::GeneralExpression, x2::Variable) = add!(promote(ex1, x2)...)
add!(x1::Variable, ex2::GeneralExpression) = add!(promote(x1, ex2)...)
+(ex1::GeneralExpression, x2::Variable) = +(promote(ex1, x2)...)
+(x1::Variable, ex2::GeneralExpression) = +(promote(x1, ex2)...)
minus!(ex1::GeneralExpression, x2::Variable) = minus!(promote(ex1, x2)...)
minus!(x1::Variable, ex2::GeneralExpression) = minus!(promote(x1, ex2)...)
-(ex1::GeneralExpression, x2::Variable) = -(promote(ex1, x2)...)
-(x1::Variable, ex2::GeneralExpression) = -(promote(x1, ex2)...)
mul!(ex1::GeneralExpression, x2::Variable) = mul!(promote(ex1, x2)...)
mul!(x1::Variable, ex2::GeneralExpression) = mul!(promote(x1, ex2)...)
*(ex1::GeneralExpression, x2::Variable) = *(promote(ex1, x2)...)
*(x1::Variable, ex2::GeneralExpression) = *(promote(x1, ex2)...)
rdiv!(ex1::GeneralExpression, x2::Variable) = mul!(ex1, x2 ^ -1.0)
rdiv!(x1::Variable, ex2::GeneralExpression) = rdiv!(promote(x1, ex2)...)
/(ex1::GeneralExpression, x2::Variable) = ex1 * (x2 ^ -1.0)
/(x1::Variable, ex2::GeneralExpression) = /(promote(x1, ex2)...)

# these can all be in-place because the result of convert from
# Variable to GeneralExpression does not need to be kept around
add!(x1::Variable, x2::Variable) = add!(promote(x1, x2)...)
+(x1::Variable, x2::Variable) = add!(promote(x1, x2)...)
minus!(x1::Variable, x2::Variable) = minus!(promote(x1, x2)...)
-(x1::Variable, x2::Variable) = minus!(promote(x1, x2)...)
mul!(x1::Variable, x2::Variable) = mul!(promote(x1, x2)...)
*(x1::Variable, x2::Variable) = mul!(promote(x1, x2)...)
rdiv!(x1::Variable, x2::Variable) = rdiv!(promote(x1, x2)...)
/(x1::Variable, x2::Variable) = rdiv!(promote(x1, x2)...)
add!(x1::Variable, v2::Real) = add!(convert(GeneralExpression, x1), v2)
add!(v1::Real, x2::Variable) = add!(convert(GeneralExpression, x2), v1)
+(x1::Variable, v2::Real) = add!(convert(GeneralExpression, x1), v2)
+(v1::Real, x2::Variable) = add!(convert(GeneralExpression, x2), v1)
minus!(x1::Variable, v2::Real) = minus!(convert(GeneralExpression, x1), v2)
minus!(v1::Real, x2::Variable) = minus!(v1, convert(GeneralExpression, x2))
-(x1::Variable, v2::Real) = minus!(convert(GeneralExpression, x1), v2)
-(v1::Real, x2::Variable) = minus!(v1, convert(GeneralExpression, x2))
mul!(x1::Variable, v2::Real) = scale!(convert(GeneralExpression, x1), v2)
mul!(v1::Real, x2::Variable) = scale!(convert(GeneralExpression, x2), v1)
*(x1::Variable, v2::Real) = scale!(convert(GeneralExpression, x1), v2)
*(v1::Real, x2::Variable) = scale!(convert(GeneralExpression, x2), v1)
rdiv!(x1::Variable, v2::Real) = scale!(convert(GeneralExpression, x1), 1.0 / v2)
rdiv!(v1::Real, x2::Variable) = scale!(x2 ^ -1.0, v1)
/(x1::Variable, v2::Real) = scale!(convert(GeneralExpression, x1), 1.0 / v2)
/(v1::Real, x2::Variable) = scale!(x2 ^ -1.0, v1)

^(x1::Variable, v2::Integer) = x1 ^ float64(v2)
function ^(x1::Variable, v2::Real)
    model = x1.model
    numvars = model.numvars
    return GeneralExpression(model, [1.0], sparsevec(x1.idx, v2, numvars),
        v2 >= minfunctioncode, spzeros(0, 0), spzeros(numvars, 0))
end
for (f, code) in functioncodes
    f == :minval && continue
    @eval (Base.$f)(x1::Variable) = x1 ^ $code
end
