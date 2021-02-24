def setcas(nx, lx, s, v):
    k = 1
    l = 0
    delta = lx * 0.01
    while abs(l - lx) > delta:
        if k <= 2:
            k += 0.001
        else:
            print('Невозможно разбить сетку!')
            raise Exception()
        n = round((nx - s) / 2)
        l = v * s + 2 * (v * k * (k ** n - 1) / (k - 1))
        if abs(l - lx) < delta:
            x = []
            for i in range(1, n + 1):
                x.append(round(v * k ** i, 3))
            x = x[::-1] + [v] * s + x
            return x

if __name__ == "__main__":
    result = setcas(60, 1000, 10, 0.005)

    print(result)