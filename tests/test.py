def f1(x):
    print(x)


def f2(x, **kwargs):
    print(kwargs)
    print(type(kwargs))
    print(x)


f2(3)
