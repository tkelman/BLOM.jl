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

function num2expr(v1::Number, model::Model)
    numvars = model.numvars
    return GeneralExpression(model, Float64[v1], spzeros(numvars, 1),
        false, spzeros(0, 0), spzeros(numvars, 0))
end

add!(ex1::GeneralExpression, v2::Number) = add!(ex1, num2expr(v2, ex1.model))
add!(v1::Number, ex2::GeneralExpression) = add!(ex2, num2expr(v1, ex2.model))
+(ex1::GeneralExpression, v2::Number) = ex1 + num2expr(v2, ex1.model)
+(v1::Number, ex2::GeneralExpression) = ex2 + num2expr(v1, ex2.model)
minus!(ex1::GeneralExpression, v2::Number) = add!(ex1, num2expr(-v2, ex1.model))
minus!(v1::Number, ex2::GeneralExpression) =
    add!(mul!(ex2, -1.0), num2expr(v1, ex2.model))
-(ex1::GeneralExpression, v2::Number) = ex1 + num2expr(-v2, ex1.model)
-(v1::Number, ex2::GeneralExpression) = num2expr(v1, ex2.model) - ex2

function mul!(ex1::GeneralExpression, v2::Number)
    scale!(ex1.coefs, v2)
    return ex1
end

function mul!(v1::Number, ex2::GeneralExpression)
    scale!(ex2.coefs, v1)
    return ex2
end

function *(ex1::GeneralExpression, v2::Number)
    return GeneralExpression(ex1.model, ex1.coefs .* v2, copy(ex1.exponents),
        ex1.specialfcn, copy(ex1.auxK), copy(ex1.auxPt))
end

function *(v1::Number, ex2::GeneralExpression)
    return GeneralExpression(ex2.model, v1 .* ex2.coefs, copy(ex2.exponents),
        ex2.specialfcn, copy(ex2.auxK), copy(ex2.auxPt))
end

