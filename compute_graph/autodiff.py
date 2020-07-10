# -*- coding:utf-8 _*-

"""
@author: yufu
@file: autodiff
@time: 2019/10/15
"""

import numpy as np

_name_counts = {}
_global_scope = {}


class BaseOperation:
    
    def __init__(self, name="BaseOperation"):
        if name in _global_scope:
            _name_counts[name] += 1
            name = name + "_" + str(_name_counts[name])
        else:
            _name_counts[name] = 0
        _global_scope[name] = self
        self.name = name
    
    def set_name(self, name):
        self.name = name
    
    def __add__(self, other):
        return Add(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return Subtract(self, other)
    
    def __rsub__(self, other=0):
        return Subtract(other, self)
    
    def __isub__(self, other):
        return self.__sub__(other)
    
    def __neg__(self):
        return Subtract(0, self)
    
    def __mul__(self, other):
        return Multiple(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, power, modulo=None):
        return Power(self, power)
    
    def __truediv__(self, other):
        return Divide(self, other)
    
    def __rdiv__(self, other):
        return Divide(other, self)
    
    def __idiv__(self, other):
        return self.__truediv__(other)
    
    def _grad(self, name=None):
        pass
    
    def grad(self, name=None):
        if isinstance(name, BaseOperation):
            name = name.name
        if name == self.name:
            return 1
        return self._grad(name)


class Operation(BaseOperation):
    
    def __init__(self, x=None, y=None, name="Operation"):
        super(Operation, self).__init__(name)
        if x is not None:
            if isinstance(x, BaseOperation):
                self._op1 = x
            else:
                self._op1 = Constant(x)
        if y is not None:
            if isinstance(y, BaseOperation):
                self._op2 = y
            else:
                self._op2 = Constant(y)


class Constant(BaseOperation):
    
    def __init__(self, x, name="Constant"):
        super(Constant, self).__init__(name)
        self._value = x
    
    def get_value(self):
        return self._value
    
    def forward(self):
        return self._value
    
    def _grad(self, name=None):
        return 0


class Variable(BaseOperation):
    
    def __init__(self, value, name="Variable"):
        super(Variable, self).__init__(name)
        self._value = value
    
    def get_value(self):
        return self._value
    
    def set_value(self, value):
        self._value = value
    
    def forward(self):
        return self._value
    
    def _grad(self, name=None):
        if name is None or name == self.name:
            return 1
        return 0


class Add(Operation):
    
    def __init__(self, x, y, name="Add"):
        super(Add, self).__init__(x, y, name)
    
    def forward(self):
        return self._op1.forward() + self._op2.forward()
    
    def _grad(self, name=None):
        return self._op1.grad(name) + self._op2.grad(name)


class Subtract(Operation):
    
    def __init__(self, x, y, name="Subtract"):
        super(Subtract, self).__init__(x, y, name)
    
    def forward(self):
        return self._op1.forward() - self._op2.forward()
    
    def _grad(self, name=None):
        return self._op1.grad(name) - self._op2.grad(name)


class Multiple(Operation):
    
    def __init__(self, x, y, name="Multiple"):
        super(Multiple, self).__init__(x, y, name)
    
    def forward(self):
        return self._op1.forward() * self._op2.forward()
    
    def _grad(self, name=None):
        return self._op1.forward() * self._op2.grad(name) + self._op2.forward() * self._op1.grad(name)


class Power(Operation):
    
    def __init__(self, x, power, name="Power"):
        super(Power, self).__init__(x, name=name)
        self._power = power
    
    def forward(self):
        return self._op1.forward() ** self._power
    
    def _grad(self, name=None):
        return self._power * (self._op1.forward() ** (self._power - 1)) * self._op1.grad(name)


class Divide(Operation):
    
    def __init__(self, x, y, name="Divide"):
        super(Divide, self).__init__(x, y, name)
    
    def forward(self):
        return self._op1.forward() / self._op2.forward()
    
    def _grad(self, name=None):
        x = self._op1.forward()
        y = self._op2.forward()
        dx = self._op1.grad(name)
        dy = self._op2.grad(name)
        return (dx * y - dy * x) / y ** 2


class Log(Operation):
    
    def __init__(self, x, name="Log"):
        super(Log, self).__init__(x, name=name)
    
    def forward(self):
        return np.log(self._op1.forward())
    
    def _grad(self, name=None):
        return 1 / self._op1.forward() * self._op1.grad(name)


class Exp(Operation):
    
    def __init__(self, x, name="Exp"):
        super(Exp, self).__init__(x, name=name)
    
    def forward(self):
        return np.exp(self._op1.forward())
    
    def _grad(self, name=None):
        return self.forward() * self._op1.grad(name)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid(Operation):
    
    def __init__(self, x, name="Sigmoid"):
        super(Sigmoid, self).__init__(x, name=name)
    
    def forward(self):
        return _sigmoid(self._op1.forward())
    
    def _grad(self, name=None):
        x = self._op1.forward()
        return _sigmoid(x) * (1 - _sigmoid(x)) * self._op1.grad(name)


class Tanh(Operation):
    
    def __init__(self, x, name="Tanh"):
        super(Tanh, self).__init__(x, name=name)
    
    def forward(self):
        return np.tanh(self._op1.forward())
    
    def _grad(self, name=None):
        return (1 - np.tanh(self._op1.forward()) ** 2) * self._op1.grad(name)


class Relu(Operation):
    
    def __init__(self, x, name="Relu"):
        super(Relu, self).__init__(x, name=name)
    
    def forward(self):
        x = self._op1.forward()
        if x <= 0:
            return 0
        return x
    
    def _grad(self, name=None):
        if self._op1.forward() <= 0:
            return 0
        return self._op1.grad(name)


class LeakyRelu(Operation):
    
    def __init__(self, x, coef=0.1, name="LeakyRelu"):
        super(LeakyRelu, self).__init__(x, name=name)
        self._coef = coef
    
    def forward(self):
        x = self._op1.forward()
        if x <= 0:
            return self._coef * x
        return x
    
    def _grad(self, name=None):
        if self._op1.forward() <= 0:
            return self._coef * self._op1.grad(name)
        return self._op1.grad(name)
