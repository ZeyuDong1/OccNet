import numpy as np
from mayavi import mlab
import os 

def visual_flow_grid(root_path,type_num,epoch,iter):
    # root_path = '/mnt/sata4t/Projects/FBOCC/FB-BEV/flowsPre'
    types = ["gt","output"]
    type=types[type_num]
    filename= f"flow_{type}_{epoch}_{iter}.npy"

    flow_path = os.path.join(root_path,filename)
    flow_grid=np.load(flow_path)
    flow_grid = flow_grid[0]
    if type == "output":
        flow_grid = flow_grid.transpose(1,2,3,0)
    
    x, y, z = np.mgrid[0:200, 0:200, 0:16]  # 根据flow的形状创建网格，这里可能需要根据实际情况调整
    u, v, w = flow_grid[..., 0], flow_grid[..., 1], flow_grid[..., 2]  # 提取矢量分量
    # 使用quiver3d函数进行可视化
    mlab.quiver3d(x, y, z, u, v, w ,scale_mode = "vector",colormap='rainbow')


if __name__ == '__main__':
    root_path = './data'
    # visual_flow_grid(root_path,0,4,0)
    visual_flow_grid(root_path,1,4,0)
    
    # 显示图形
    mlab.show()