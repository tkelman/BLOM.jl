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

function add!(ex1::GeneralExpression, v2::Real)
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
-(v1::Real, ex2::GeneralExpression) = add!(scale!(copy(ex2), -1.0), v1)

function *(ex1::GeneralExpression, v2::Real)
    return GeneralExpression(ex1.model, ex1.coefs .* v2, copy(ex1.exponents),
        ex1.specialfcn, copy(ex1.auxK), copy(ex1.auxPt))
end

function *(v1::Real, ex2::GeneralExpression)
    return GeneralExpression(ex2.model, v1 .* ex2.coefs, copy(ex2.exponents),
        ex2.specialfcn, copy(ex2.auxK), copy(ex2.auxPt))
end

