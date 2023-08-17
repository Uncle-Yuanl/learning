print('<[100]> evalsupport module start')

def deco_alpha(cls):
    print('<[200]> deco_alpha')

    def inner_1(self):
        print('<[300]> deco_alpha:inner_1')

    cls.method_y = inner_1
    return cls


def deco_func(func):
    print('<[220]> deco_func: ', func.__name__)

    def inner_2():
        print('<[222]> deco_func:inner_2')

    return inner_2


def norm_func():
    print('<[230]> norm_func')


class MetaAleph(type):
    print('<[400]> MetaAleph body')

    def __init__(cls, name, bases, dic):
        print('<[500]> MetaAleph.__init__')

        def inner_2(self):
            print('<[600]> MetaAleph.__init__:inner_2')

        cls.method_z = inner_2

    
class NormClass:

    print("<[440]> Normal Class")


print('<[700]> evalsupport module end')