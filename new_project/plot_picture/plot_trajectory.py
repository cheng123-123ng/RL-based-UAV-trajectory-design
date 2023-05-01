from plot_picture.render import render
import numpy as np

f = 'D:\\桌面\\课程\\毕业设计\\毕业论文\\new_project\\result\\trajectory_1uav_2ucar_train50{}.npy'
trajectory = np.load(f,allow_pickle=True).item()
render(trajectory,3,f'D:\\桌面\\课程\\毕业设计\\毕业论文\\照片与数据集\\trajectory_1uav_2ucar_train50.gif')