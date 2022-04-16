import numpy as np
from sklearn.cross_decomposition import CCA

def sym_sqrt(A):
    a, b = np.linalg.eig(A)
    return np.dot(b, np.dot(np.diag(np.sqrt(a)), b.T))

def test_sym_sqrt():
    x = np.random.normal(size=(3, 3))
    x = np.dot(x.T, x)
    s = sym_sqrt(x)
    assert((abs(s - s.T) < 1e-8).all())
    assert((abs(x - np.dot(s, s)) < 1e-8).all())

def gen_fungrad(X, Y, c1, c2):

    n, px = X.shape
    py = Y.shape[1]
    assert X.shape[0] == Y.shape[0]

    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = X.shape[0]

    Sxx = np.dot(X.T, X) / n
    Rx = sym_sqrt(Sxx)
    Xt = np.linalg.solve(Rx, X.T).T
    Syy = np.dot(Y.T, Y) / n
    Ry = sym_sqrt(Syy)
    Yt = np.linalg.solve(Ry, Y.T).T

    assert np.allclose(np.dot(Xt.T, Xt) / Xt.shape[0], np.eye(Xt.shape[1]))
    assert np.allclose(np.dot(Yt.T, Yt) / Yt.shape[0], np.eye(Yt.shape[1]))

    # Confirm that \|b\|^2 = btilde * Sxx^{-1} btilde.
    b = np.random.normal(size=X.shape[1])
    bt = np.dot(Rx, b)
    assert np.allclose(np.linalg.norm(b)**2, 
                       np.dot(bt, np.linalg.solve(Sxx, bt)))

    StXY = np.dot(Xt.T, Yt) / n
    
    def fun(x):
        assert len(x) == px + py
        a = x[0:px]
        b = x[px:px+py]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)

        # The log of the CCA objective
        f = np.dot(a, np.dot(StXY, b))
        if f <= 0:
            return -np.Inf
        f = np.log(f) - np.log(na) - np.log(nb)

        # The logs of the penalty functions
        f += c1*(2*np.log(na) - np.log(np.dot(a, np.linalg.solve(Sxx, a))))
        f += c2*(2*np.log(nb) - np.log(np.dot(b, np.linalg.solve(Syy, b))))

        return f

    def grad(x):
        a = x[0:px]
        b = x[px:px+py]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)

        # The gradient of the CCA objective
        f = np.dot(a, np.dot(StXY, b))
        if f <= 0:
            return np.zeros(px + py)
        ga = np.dot(StXY, b) / f - a/na**2 
        gb = np.dot(StXY.T, a) / f - b/nb**2

        # The gradient of the left side penalty
        ai = np.linalg.solve(Sxx, a)
        ha = np.dot(a, ai)
        ga += 2*c1*(a/na**2 - ai/ha)

        # The gradient of the right side penalty
        bi = np.linalg.solve(Syy, b)
        hb = np.dot(b, bi)
        gb += 2*c2*(b/nb**2 - bi/hb)

        return np.concatenate((ga, gb))

    return fun, grad, Xt, Yt, Rx, Ry


def optim(X, Y, c1, c2, maxiter=5000, gtol=1e-3):

    fun, grad, Xt, Yt, Rx, Ry = gen_fungrad(X, Y, c1, c2)

    px, py = X.shape[1], Y.shape[1]

    # Starting value
    r = CCA(n_components=1)
    r.fit(Xt, Yt)
    z = np.concatenate((r.x_loadings_[:,0], r.y_loadings_[:,0]))

    f0 = fun(z)
    success = False
    for iter in range(maxiter):

        g = grad(z)
        if np.linalg.norm(g) < gtol:
            success = True
            break

        step = 1.0
        success1 = False
        while step > 1e-15:
            z1 = z + step * g
            f1 = fun(z1)
            if f1 > f0:
                f0 = f1
                z = z1
                success1 = True
                break
            step /= 2
        if not success1:
            break

    if not success:
        print("|grad|=%f" % np.linalg.norm(g))

    # Map back to original coordinates and rescale
    a = z[0:px]
    b = z[px:]
    a = np.linalg.solve(Rx.T, a)
    b = np.linalg.solve(Ry.T, b)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    return a, b, success


def genar(n, p, r):
    X = np.random.normal(size=(n, p))
    for j in range(1, p):
        X[:, j] = r*X[:, j-1] + np.sqrt(1 - r**2)*X[:, j]
    return X


def test_fungrad():
    n = 1000
    px = 3
    py = 3
    r = 0.4

    # Generate X
    X = genar(n, px, r)

    # Generate Y
    Y = genar(n, py, r)

    fun, grad, _, _, _, _ = gen_fungrad(X, Y, 3, 2)

    for k in range(20):
        z = np.random.normal(size=px+py)
        if fun(z) == -np.Inf:
            z[0:px] *= -1

        ngrad = np.zeros(px+py)
        f0 = fun(z)
        ee = 1e-8
        for k in range(len(z)):
            z0 = z.copy()
            z0[k] += ee
            f1 = fun(z0)
            ngrad[k] = (f1 - f0) / ee
        g = grad(z)
        assert np.allclose(g, ngrad, rtol=1e-4, atol=1e-4)

def test_optim():

    n = 1000
    px = 30
    py = 30
    r = 0.8

    X = genar(n, px, r)
    Y = genar(n, py, r)
    Y[:, 1] = r*X[:, 0] + np.sqrt(1 - r**2)*np.random.normal(size=n)

    r = CCA(n_components=1)
    r.fit(X, Y)
    Xc, Yc = r.transform(X, Y)
    r0 = np.corrcoef(Xc[:, 0], Yc[:, 0])[0, 1]

    for f in [0, 1, 2, 3]:
        print(f)
        a, b, success = optim(X, Y, f, 2*f, maxiter=4000)
        r1 = np.corrcoef(np.dot(X, a), np.dot(Y, b))[0, 1]
        if f == 0:
            assert np.allclose(r0, r1)
        assert success

test_fungrad()
test_optim()
