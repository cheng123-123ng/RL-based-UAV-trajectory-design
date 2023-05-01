from plot_picture.render_1uav_1ED import render
import numpy as np

f = f'D:\\桌面\\课程\\毕业设计\\毕业论文\\照片与数据集\\trajectory_1uav_1ED_2ucar_train50.gif'
trajectory = np.load('D:\\桌面\\课程\\毕业设计\\毕业论文\\new_project\\result\\trajectory_1uav_1ED_2Ucar_train50.npy',allow_pickle=True).item()
render(trajectory,4,f)