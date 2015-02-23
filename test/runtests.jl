using BLOM
using Base.Test

model = BLOM.Model()
x1 = BLOM.newVariable(model)
x2 = BLOM.newVariable(model)
x3 = BLOM.newVariable(model)

ex1 = convert(BLOM.GeneralExpression, x3)
ex2 = convert(BLOM.GeneralExpression, x1)

ex1 + ex2
