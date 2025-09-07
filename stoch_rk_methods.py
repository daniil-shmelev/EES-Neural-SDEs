from stoch_rk import StochRK, from_kauri
import kauri as kr

heun_rk3 = StochRK([[0, 0, 0],
                        [1/3, 0, 0],
                        [0, 2/3, 0]],
              [1/4, 0, 3/4], 'Heun RK3')

def EES25(x):
    return from_kauri(kr.EES25(x))

def EES25_sym(x):
    method = kr.EES25(x)
    method = method * method.reverse()
    method.name = "EES25_sym"
    return from_kauri(method)

def EES27(x):
    return from_kauri(kr.EES27(x))

def EES27_sym(x):
    method = kr.EES27(x)
    method = method * method.reverse()
    method.name = "EES27_sym"
    return from_kauri(method)

RK4 = from_kauri(kr.rk4)