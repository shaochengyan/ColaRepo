import taichi as ti
import numpy as np


@ti.data_oriented
class NeighborhoodSearchSparse:
    def __init__(self,positions,particle_max_num, support_radius, domain_size, use_sparse_grid=False):
        self.particle_max_num = particle_max_num  # point number
        self.support_radius = support_radius  # query radiu
        self.domain_size = domain_size  # 3D space and maximum domain of your points set.

        self.positions = positions  # pts: vector filed 

        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        self.grid_num_1d = self.grid_num[0] * self.grid_num[1] * self.grid_num[2]
        self.dim = 3
        self.max_num_neighbors = 60  # 为了搜索时不使用 dynamic vector
        self.max_num_particles_in_grid = 50 

        self.neighbors = ti.field(int, shape=(self.particle_max_num, self.max_num_neighbors))  # Neighbors of each point
        self.num_neighbors = ti.field(int, shape=self.particle_max_num)  # neighbors number of each point

        if not use_sparse_grid:  # 将空间划分为 D1xD2XD3, 然后每个格子下存储最多 max_num_particles_in_grid 个点
            self.grid_particles_num = ti.field(int, shape=(self.grid_num))  # 每个格子下的点的数量
            self.particles_in_grid = ti.field(int, shape=(*self.grid_num, self.max_num_particles_in_grid))  # 每个格子下点的坐标!
        else:  # 稀疏存储方式!  TODO: 
            self.particles_in_grid = ti.field(int)
            self.grid_particles_num = ti.field(int)
            self.grid_snode = ti.root.bitmasked(ti.ijk, self.grid_num)  # 创建格子网格
            self.grid_snode.place(self.grid_particles_num) 
            self.grid_snode.bitmasked(ti.l, self.max_num_particles_in_grid).place(self.particles_in_grid)
        
        # COLA:
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
    def update_grid(self):
        for i in range(self.particle_max_num):
            grid_index = self.pos_to_index(self.positions[i])
            k = ti.atomic_add(self.grid_particles_num[grid_index], 1)  # number++
            self.particles_in_grid[grid_index, k] = i  # append unmber

    def run_search(self):
        self.num_neighbors.fill(0)
        self.neighbors.fill(-1)
        self.particles_in_grid.fill(-1)
        self.grid_particles_num.fill(0)

        self.update_grid()  # pts to sparse grid

        self.store_neighbors()  # search each point in other point
        # print("Grid usage: ", self.grid_usage())

    @ti.func
    def is_in_grid(self, c):
        return 0 <= c[0] < self.grid_num[0] and 0 <= c[1] < self.grid_num[1] and 0 <= c[2] < self.grid_num[2]

    @ti.kernel
    def store_neighbors(self):
        for p_i in range(self.particle_max_num):  
            center_cell = self.pos_to_index(self.positions[p_i])  # idx of query point
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):  # 3X3X3 neighbor grids
                grid_index = center_cell + offset  # grids on
                if self.is_in_grid(grid_index):  # check boundary
                    for k in range(self.grid_particles_num[grid_index]):  # each point -> check distant
                        p_j = self.particles_in_grid[grid_index, k]
                        if p_i != p_j and (self.positions[p_i] - self.positions[p_j]).norm() < self.support_radius:
                            kk = ti.atomic_add(self.num_neighbors[p_i], 1)
                            self.neighbors[p_i, kk] = p_j


    @ti.kernel
    def run_query(self):
        for p_i in range(self.num_queries):  
            center_cell = self.pos_to_index(self.queries[p_i])  # idx of query point
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):  # 3X3X3 neighbor grids
                grid_index = center_cell + offset  # grids on
                if self.is_in_grid(grid_index):  # check boundary
                    for k in range(self.grid_particles_num[grid_index]):  # each point -> check distant
                        p_j = self.particles_in_grid[grid_index, k]
                        if p_i != p_j and (self.positions[p_i] - self.positions[p_j]).norm() < self.support_radius:
                            kk = ti.atomic_add(self.neighbors_num[p_i], 1)
                            self.neighbors_idx[p_i, kk] = p_j


    def search_neighbors(self, queries_np):
        """
        querys: Nx3 ndarray float32
        """
        # to filed like postion
        self.num_queries = len(queries_np)
        self.queries = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_queries))
        self.queries.from_numpy(queries_np)

        # rslt: numbers and idx of neighbors
        self.neighbors_idx = ti.field(int, shape=(self.num_queries, self.max_num_neighbors))  # Neighbors of each point
        self.neighbors_num = ti.field(int, shape=self.num_queries)  # neighbors number of each point

        # init
        self.particles_in_grid.fill(-1)
        self.grid_particles_num.fill(0)
        self.update_grid()  # pts to sparse grid

        self.neighbors_idx.fill(-1)
        self.neighbors_num.fill(0)
        
        # run search
        self.run_query()
        
        # return to np
        # print(self.neighbors)
        # print(self.neighbors_num)

        return self.neighbors_idx.to_numpy(), self.neighbors_num.to_numpy()
        