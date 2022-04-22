import jax
import jax.numpy as np


def load_face(idx=2):
    '''
    Loads a face mesh from http://tosca.cs.technion.ac.il/book/resources_data.html
    '''
    import numpy as np_

    tri = np_.loadtxt(f'face/face{idx:02}.tri', dtype=np_.int32) - 1
    N = tri.shape[0]
    assert tri.shape == (N, 3)

    vert = np_.loadtxt(f'face/face{idx:02}.vert')
    assert vert.shape == (tri.max() + 1, 3)
    vert = vert - vert.mean(axis=0)
    vert = vert / np.abs(vert).max()

    mesh = np.array(vert[tri])
    assert mesh.shape == (N, 3, 3)

    text = np_.loadtxt(f'face/face{idx:02}.text')
    text = (text[tri[:, 0]] + text[tri[:, 1]] + text[tri[:, 2]]) / 3
    assert text.shape == (N,)
    
    return mesh, text

def load_pizza():
    mesh = np.array([
        [ [-0.5, -0.8, -0.2], [+0.5, -0.8, -0.2], [ 0  , +0.8, +0.2] ],
        [ [-0.5, +0.8, -0.2], [+0.5, +0.8, -0.2], [ 0  , -0.8, +0.2] ]
    ])
    text = np.array([1., 0.5])
    return mesh, text

def get_pixel(left=-1, top=1, right=1.5, bottom=-1.5, xres=50, yres=50):
    '''
    Generates a grid of pixel samples in a given viewport, at a given resolution
    '''
    Xs = np.linspace(left, right, xres)
    Ys = np.linspace(top, bottom, yres)
    X, Y = np.meshgrid(Xs, Ys)
    Z = np.zeros_like(X)
    pixel = np.stack((X, Y, Z), axis=2).reshape(-1, 1, 3)
    return pixel, Z.shape

eps = 1e-8

@jax.jit
def softras(proj, pixel, C, SIGMA=1e-1, GAMMA=1e-1):
    Zbuf = proj[:, :, 2]
    proj = proj.at[:, :, 2].set(0)
    
    def dot(a, b):
        return (a * b).sum(axis=-1, keepdims=True)

    def d2_point_to_finite_edge(i):
        A = proj[:, i, :]
        B = proj[:, (i + 1) % 3, :]
        Va = B - A
        Vp = pixel - A
        projln = dot(Vp, Va) / (dot(Va, Va) + eps)
        projpt = np.clip(projln, 0, 1) * Va[None, :, :]
        out = dot(Vp - projpt, Vp - projpt)
        return out[:, :, 0]

    d2 = np.minimum(
        np.minimum(d2_point_to_finite_edge(0), d2_point_to_finite_edge(1)),
        d2_point_to_finite_edge(2)
    )
    
    def signed_area_to_point(i):
        A = proj[:, i, :]
        B = proj[:, (i + 1) % 3, :]
        Va = B - A
        area = np.cross(Va, pixel - A)[:, :, 2] / 2
        return area

    Aa = signed_area_to_point(0)
    Ab = signed_area_to_point(1)
    Ac = signed_area_to_point(2)
    Aabc = Aa + Ab + Ac + eps
    in_triangle =\
        np.equal(np.sign(Aa), np.sign(Ab)).astype('float32') *\
        np.equal(np.sign(Aa), np.sign(Ac)).astype('float32') * 2 - 1

    D = jax.nn.sigmoid(in_triangle * d2 / SIGMA)

    bary = np.stack([Aa, Ab, Ac], axis=2) / Aabc[:, :, None]
    bary_clipped = np.clip(bary, 0, 1)
    bary_clipped = bary_clipped / (bary_clipped.sum(axis=2, keepdims=True) + eps)

    Zb = (bary_clipped * np.roll(Zbuf, 2, axis=1)).sum(axis=2)
    Zb = (Zb.max() - Zb) / (Zb.max() - Zb.min())

    Zbe = np.exp(np.clip(Zb / GAMMA, -20., 20.))
    DZbe = D * Zbe
    w = DZbe / (DZbe.sum(axis=1, keepdims=True) + np.exp(eps / GAMMA))
    return (w * C).sum(axis=1)


def make_rotation_Y(theta):
    return np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)]
    ])

def apply_homogeneous_transform(T, mesh):
    return np.matmul(T, mesh[:, :, :, None])[:, :, :, 0]

def apply_translation(mesh, dx, dy, dz):
    return mesh + np.stack([dx, dy, dz])