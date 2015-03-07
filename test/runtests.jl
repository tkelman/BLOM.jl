using BLOM
using Base.Test

model = BLOM.Model()
x1 = BLOM.Variable(model, start = 1.0)
x2 = BLOM.Variable(model)
x3 = BLOM.Variable(model)

ex1 = convert(BLOM.GeneralExpression, x3)
ex2 = convert(BLOM.GeneralExpression, x1)

@test getValue(sin(x1)) == sin(1.0)
