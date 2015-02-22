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
