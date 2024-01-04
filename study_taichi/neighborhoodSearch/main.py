# this is a test case for neighborhood search
import plyfile
import numpy as np
import taichi as ti

from neighborhood_search import NeighborhoodSearchSparse


def read_ply_particles(geometryFile):
    plydata = plyfile.PlyData.read(geometryFile)
    pts = np.stack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]], axis=1)
    return pts

def test_small():
    '''
    domain size = 1.0, 1.0, 1.0
    Sparse grid may not save memory
    '''
    ti.init(arch=ti.gpu, device_memory_GB=0.15)
    pts = read_ply_particles("cube.ply")
    particle_max_num = pts.shape[0]
    positions = ti.Vector.field(3, dtype=ti.f32, shape=particle_max_num)
    positions.from_numpy(pts)
    domain_size = np.array([1.0, 1.0, 1.0])
    ns = NeighborhoodSearchSparse(positions, particle_max_num, 0.04, domain_size, use_sparse_grid=False)
    # if we change use_sparse_grid to True, the program will crash, because sparse grid does not save memory, instead, it cost more memory
    ns.run_search()
    np.savetxt("neighbors.txt", ns.neighbors.to_numpy(), fmt="%d")
    np.savetxt("num_neighbors.txt", ns.num_neighbors.to_numpy(), fmt="%d")
    print("small domain test done")

def test_large():
    '''
    domain size = 10.0, 10.0, 10.0
    Sparse grid may save memory
    '''
    ti.init(arch=ti.gpu, device_memory_GB=3.3)
    pts = read_ply_particles("cube.ply")
    particle_max_num = pts.shape[0]  # 1w...
    # numpy points -> ti vector filed
    positions = ti.Vector.field(3, dtype=ti.f32, shape=particle_max_num)
    positions.from_numpy(pts)

    domain_size = np.array([10.0, 10.0, 10.0])

    ns = NeighborhoodSearchSparse(positions, particle_max_num, 0.04, domain_size, use_sparse_grid=True)

    # if we change use_sparse_grid to False, the program will crash, because sparse grid does save memory.
    ns.run_search()
    np.savetxt("neighbors.txt", ns.neighbors.to_numpy(), fmt="%d")
    np.savetxt("num_neighbors.txt", ns.num_neighbors.to_numpy(), fmt="%d")
    print("Grid usage: ", ns.grid_usage())
    print("large domain test done")


# COLA TEST:
def test_radiu_search():
    ti.init(arch=ti.gpu, device_memory_GB=3.3)

    from radiu_search import RadiuSearchSparse

    # pts = read_ply_particles("cube.ply")
    pts = np.random.random(size=(20000, 3)).astype(np.float32)

    from utils import Timer

    N = 10000
    T = 20
    # domain_size = np.array([10.0, 10.0, 10.0])
    domain_size = np.ones(3) * 3
    rss = RadiuSearchSparse(points=pts, query_radius=0.04, domain_size=domain_size, use_sparse_grid=True)
    with Timer("Taichi"):
        for i in range(T):
            a, b = rss.search_neighbors(queries_np=pts[:N])
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(radius=0.04)
    nn.fit(pts)
    with Timer("Sklearn"):
        for i in range(T):
            rslt = nn.radius_neighbors(pts[:N])

    from sklearn.neighbors import KDTree
    nn = KDTree(pts)
    with Timer("Sklearn: KDTree"):
        for i in range(T):
            rslt = nn.query_radius(pts[:N], r=0.04)


if __name__ == '__main__':
    # test_small()
    # test_large()
    test_radiu_search()