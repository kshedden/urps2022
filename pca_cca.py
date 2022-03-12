import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.utils import Bunch

# Generate an 'n x p' array of data with autocorrelation at level 'r'.
def genar(n, p, r):
    x = np.random.normal(size=(n, p))
    for j in range(1, p):
        x[:, j] = r*x[:, j-1] + np.sqrt(1 - r**2)*x[:, j]
    return x

# Use PCA to reduce the 'x' data to 'kx' variates, and to reduce the
# 'y' data to 'ky' variates, then use CCA to reduce the PC scores to
# 'qq' variates.
def pca_cca(x, y, kx, ky, qq):

    # Center the variables
    x = x - x.mean(0)
    y = y - y.mean(0)

    # Reduce X using PCA
    xpca = PCA(kx)
    xpca.fit(x)
    xpcaproj = xpca.components_.T
    xpcascores = xpca.transform(x)

    # Reduce Y using PCA
    ypca = PCA(ky)
    ypca.fit(y)
    ypcaproj = ypca.components_.T
    ypcascores = ypca.transform(y)

    # Conduct CCA
    cca = CCA(qq, scale=False)
    cca.fit(xpcascores, ypcascores)
    xccaproj = cca.x_rotations_
    yccaproj = cca.y_rotations_
    xccascores, yccascores = cca.transform(xpcascores, ypcascores)

    # Get the canonical correlations
    cancor = [np.corrcoef(xccascores[:, j], yccascores[:, j])[0, 1] for j in range(qq)]
    cancor = np.asarray(cancor)

    # Linear map from the original variables to the PCA/CCA reducted variables
    xfullmap = np.dot(xpcaproj, xccaproj)
    xfullmap /= np.linalg.norm(xfullmap, axis=0)
    yfullmap = np.dot(ypcaproj, yccaproj)
    yfullmap /= np.linalg.norm(yfullmap, axis=0)

    return Bunch(xc=x, yc=y, cancor=cancor,
                 xfullmap=xfullmap, yfullmap=yfullmap,
                 xpcascores=xpcascores, xccascores=xccascores,
                 xpcaproj=xpcaproj, xccaproj=xccaproj,
                 ypcascores=ypcascores, yccascores=yccascores,
                 ypcaproj=ypcaproj, yccaproj=yccaproj)


# Run some checks on the PCA and CCA calculations.  Sample size (n),
# dimension of the observed data (px, py), autocorrelation for
# generated data (r), dimensions for PCA rediction (kx, ky), dimension
# of CCA reduction (qq).
def test_pca_cca(n, px, py, r, kx, ky, qq):

    x = genar(n, px, r)
    y = genar(n, py, r)

    # Induce cross-correlations
    x[:, 0] = y.sum(1)
    x[:, 0] /= x[:, 0].std()
    x[:, 0] += np.random.normal(size=n)

    r = pca_cca(x, y, kx, ky, qq)

    # Confirm that we get the same scores by direct calculation
    xpcascores1 = np.dot(r.xc, r.xpcaproj)
    assert np.allclose(r.xpcascores, xpcascores1)

    # Confirm that the PCA basis is orthogonal
    assert np.allclose(np.dot(r.xpcaproj.T, r.xpcaproj), np.eye(kx))

    # Confirm that we get the same scores by direct calculation
    ypcascores1 = np.dot(r.yc, r.ypcaproj)
    assert np.allclose(r.ypcascores, ypcascores1)

    # Confirm that the PCA basis is orthogonal
    assert np.allclose(np.dot(r.ypcaproj.T, r.ypcaproj), np.eye(ky))

    # Confirm that we get the same scores by direct calculation
    xccascores1 = np.dot(r.xpcascores, r.xccaproj)
    assert np.allclose(r.xccascores, xccascores1)
    yccascores1 = np.dot(r.ypcascores, r.yccaproj)
    assert np.allclose(r.yccascores, yccascores1)

    # Confirm that these maps perform as expected
    xccascores1 = np.dot(r.xc, r.xfullmap)
    assert np.allclose(r.xccascores, xccascores1)
    yccascores1 = np.dot(r.yc, r.yfullmap)
    assert np.allclose(r.yccascores, yccascores1)

test_pca_cca(n=1000, px=10, py=10, r=0.5, kx=3, ky=3, qq=1)


# Return the rows of pts that are on the Pareto frontier.
def find_pareto(pts):
    keep = []
    for i in range(pts.shape[0]):
        if ~(pts > pts[i, :]).all(1).any():
            keep.append(i)
    return pts[keep, :]

# Take data 'x' and 'y' (observations in rows and variables in
# columns), reduce to 'kx' and 'ky' dimensions respectively using
# PCA, then use CCA to reduce to maximally correlated variates.
# Returns 'm x 3' arrays in which each row contains a correlation, an
# x-variance, and a y-variance.  The first returned array contains all
# such points and the second returned array contains only the points
# on the Pareto front.
def pca_cca_pareto(x, y, kx, ky):

    # Center the data
    x = x - x.mean(0)
    y = y - y.mean(0)

    # sx and sy are the maximum possible variances
    # for x * u and y * v, where u and v are unit
    # vectors.
    _, sx, _ = np.linalg.svd(x, 0)
    sx = sx.max()**2 / x.shape[0]
    _, sy, _ = np.linalg.svd(y, 0)
    sy = sy.max()**2 / y.shape[0]

    pts = []
    for kx1 in range(1, kx):
        for ky1 in range(1, ky):
            r = pca_cca(x, y, kx1, ky1, 1)
            x1 = np.dot(x, r.xfullmap[:, 0])
            y1 = np.dot(y, r.yfullmap[:, 0])
            pts.append(np.r_[r.cancor[0], x1.var()/sx, y1.var()/sy])
    pts = np.vstack(pts)

    return pts, find_pareto(pts)


def test_pca_cca_pareto(n, px, py, r, kx, ky):

    x = genar(n, px, r)
    y = genar(n, py, r)

    pts, ppts = pca_cca_pareto(x, y, kx, ky)
    print(pts.shape)
    print(ppts.shape)

    pdf = PdfPages("test_pca_cca_pareto.pdf")
    vnames = ["Canonical correlation", "X variance", "Y variance"]

    for j1 in range(3):
        for j2 in range(j1+1, 3):
            plt.clf()
            plt.grid(True)
            plt.plot(pts[:, j1], pts[:, j2], "o", color="orange", mfc="none", alpha=0.5)
            plt.plot(ppts[:, j1], ppts[:, j2], "o", color="red", mfc="none")
            plt.xlabel(vnames[j1], size=15)
            plt.ylabel(vnames[j2], size=15)
            pdf.savefig()

    pdf.close()

    return pts, ppts

pts, ppts = test_pca_cca_pareto(n=1000, px=20, py=20, r=0.5, kx=20, ky=20)
