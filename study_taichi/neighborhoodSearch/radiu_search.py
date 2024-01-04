import taichi as ti
import numpy as np


@ti.data_oriented
class RadiuSearchSparse:
    def __init__(self, points, query_radius, domain_size=None, max_num_particles_in_grid=20, max_num_neighbors=20, use_sparse_grid=False):
        """
        points: ndarray numpy Nx3
        """
        # points to filed
        self.N_points = points.shape[0]
        self.points = self.ndarray_to_filed(points)

        self.domain_size = domain_size  # 3D space and maximum domain of your points set.
        if self.domain_size is None:
            self.domain_size = np.max(points, axis=0) - np.min(points, axis=0)

        self.query_radius = query_radius  # query radiu

        # grid
        self.grid_size = self.query_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        self.grid_num_1d = self.grid_num[0] * self.grid_num[1] * self.grid_num[2]
        self.dim = points.shape[1]
        self.max_num_neighbors = max_num_neighbors  # 为了搜索时不使用 dynamic vector
        self.max_num_particles_in_grid = max_num_particles_in_grid  # 一个 grid 下最多的点数 

        if not use_sparse_grid:  # 将空间划分为 D1xD2XD3, 然后每个格子下存储最多 max_num_particles_in_grid 个点
            self.grid_particles_num = ti.field(int, shape=(self.grid_num))  # 每个格子下的点的数量
            self.particles_in_grid = ti.field(int, shape=(*self.grid_num, self.max_num_particles_in_grid))  # 每个格子下点的坐标!
        else:  # 稀疏存储方式!  TODO: 
            self.particles_in_grid = ti.field(int)
            self.grid_particles_num = ti.field(int)
            self.grid_snode = ti.root.bitmasked(ti.ijk, self.grid_num)  # 创建格子网格
            self.grid_snode.place(self.grid_particles_num) 
            self.grid_snode.bitmasked(ti.l, self.max_num_particles_in_grid).place(self.particles_in_grid)
        
        self.particles_in_grid.fill(-1)
        self.grid_particles_num.fill(0)
        self.points_fill_grid()
        
        # neighbor
        self.num_queries = None
        self.queries = None

        # rslt: numbers and idx of neighbors
        self.neighbors_idx = None
        self.neighbors_num = None       

    @ti.kernel
    def grid_usage(self) -> ti.f32:
        cnt = 0
        for I in ti.grouped(self.grid_snode):
            if ti.is_active(self.grid_snode, I):
                cnt += 1
        usage = cnt / (self.grid_num_1d)
        return usage

    def deactivate_grid(self):
        self.grid_snode.deactivate_all()

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.kernel
    def points_fill_grid(self):
        for i in range(self.N_points):
            grid_index = self.pos_to_index(self.points[i])
            k = ti.atomic_add(self.grid_particles_num[grid_index], 1)  # number++
            self.particles_in_grid[grid_index, k] = i  # append unmber

    @ti.func
    def is_in_grid(self, c):
        return 0 <= c[0] < self.grid_num[0] and 0 <= c[1] < self.grid_num[1] and 0 <= c[2] < self.grid_num[2]

    @ti.kernel
    def run_query(self):
        for p_i in range(self.num_queries):  
            center_cell = self.pos_to_index(self.queries[p_i])  # idx of query point
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):  # 3X3X3 neighbor grids
                grid_index = center_cell + offset  # grids on
                if self.is_in_grid(grid_index):  # check boundary
                    for k in range(self.grid_particles_num[grid_index]):  # each point -> check distant
                        p_j = self.particles_in_grid[grid_index, k]
                        if p_i != p_j and (self.points[p_i] - self.points[p_j]).norm() < self.query_radius:
                            kk = ti.atomic_add(self.neighbors_num[p_i], 1)
                            self.neighbors_idx[p_i, kk] = p_j
    
    def ndarray_to_filed(self, ndarray):
        assert isinstance(ndarray, np.ndarray) and ndarray.ndim == 2
        assert ndarray.dtype == np.float32
        filed = ti.Vector.field(ndarray.shape[1], dtype=ti.f32, shape=(ndarray.shape[0]))
        filed.from_numpy(ndarray)
        return filed


    def search_neighbors(self, queries_np):
        """
        querys: Nx3 ndarray float32
        """
        # to filed like postion
        self.num_queries = len(queries_np)
        self.queries = self.ndarray_to_filed(queries_np)

        # rslt: numbers and idx of neighbors
        self.neighbors_idx = ti.field(int, shape=(self.num_queries, self.max_num_neighbors))  # Neighbors of each point
        self.neighbors_num = ti.field(int, shape=self.num_queries)  # neighbors number of each point

        # init
        # self.points_fill_grid()  # pts to sparse grid

        self.neighbors_idx.fill(-1)
        self.neighbors_num.fill(0)
        
        # run search
        self.run_query()
        
        return self.neighbors_idx.to_numpy(), self.neighbors_num.to_numpy()
