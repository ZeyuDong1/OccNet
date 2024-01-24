import numpy as np
from mayavi import mlab
import os 

def show_flow(root_path,type_num=0,epoch=2,iter=0):
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
    print(f"u sum:{np.sum(u)}")
    print(f"v sum:{np.sum(v)}")
    print(f"w sum:{np.sum(w)}")
    # 使用quiver3d函数进行可视化
    if type_num == 0:
        plot=mlab.quiver3d(x, y, z, u, v, w ,scale_mode = "vector",color=(0,0,1))
    else:
        plot=mlab.quiver3d(x, y, z, u, v, w ,scale_mode = "vector",color=(1,0,0))


if __name__ == '__main__':
    # root_path = '/mnt/sata4t/Projects/FBOCC/FB-BEV/flowsPre'
    # types = ["gt","output"]
    # type=types[1]
    # epoch = 1
    # iter = 4
    # filename= f"flow_{type}_{epoch}_{iter}.npy"

    # flow_path = os.path.join(root_path,filename)
    # flow_grid=np.load(flow_path)
    # flow_grid = flow_grid[0]
    # if type == "output":
    #     flow_grid = flow_grid.transpose(1,2,3,0)
    # x, y, z = np.mgrid[0:200, 0:200, 0:16]  # 根据flow的形状创建网格，这里可能需要根据实际情况调整
    # u, v, w = flow_grid[..., 0], flow_grid[..., 1], flow_grid[..., 2]  # 提取矢量分量
    # # 使用quiver3d函数进行可视化
    # plot=mlab.quiver3d(x, y, z, u, v, w ,scale_mode = "vector",colormap='rainbow')
    # # plot.module_manager.scalar_lut_manager.lut.table = colors_map
    # # plot.glyph.color_mode = 'color_by_scalar'  # 根据矢量大小着色
    epoch = 2
    iter = 5
    show_flow("/mnt/sata4t/Projects/FBOCC/FB-BEV/flowsPre",type_num=1,epoch=epoch,iter=iter)
    show_flow("/mnt/sata4t/Projects/FBOCC/FB-BEV/flowsPre",type_num=0,epoch=epoch,iter=iter)

    # 显示图形
    mlab.show() 