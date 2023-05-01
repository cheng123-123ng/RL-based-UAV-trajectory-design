# 动态画出sin函数曲线
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
def render(trajectroy=None ,f = None):

    fig, ax = plt.subplots()
    ax.grid()
    uavdata_x, uavdata_y = ([], []), ([], [])
    ucardata_x, ucardata_y = ([], []), ([], [])

    eddata_x, eddata_y = ([], []), ([], [])

    if trajectroy != None:
        uavdata_x = trajectroy["Uavx"]
        uavdata_y = trajectroy["Uavy"]

        ucardata_x = trajectroy["Ucarx"]
        ucardata_y = trajectroy["Ucary"]

        eddata_x = trajectroy["EDx"]
        eddata_y =trajectroy["EDy"]
        reward = trajectroy["reward"]
        ########uav,ucar,ed
    ln1, = ax.plot([], [], 'bo')
    ln2, = ax.plot([], [], 'co')

    ln3, = ax.plot([], [], 'r-')
    ln4, = ax.plot([], [], 'r-')

    ln5, = ax.plot([], [], 'k+')
    # ln4, = ax.plot([], [], 'g+')

    text_pt = plt.text(4, 0.8, '', fontsize=16)

    def init():
        ax.set_xlim(-5, 25)
        ax.set_ylim(-5, 25)
        # 返回曲线
        return ln1,ln2,ln3,ln4,ln5


    def update(frame):
        # 将每次传过来的n追加到xdata中
        ln1.set_data(uavdata_x[0][:frame], uavdata_y[0][:frame])
        ln2.set_data(uavdata_x[1][:frame], uavdata_y[1][:frame])

        ln3.set_data(ucardata_x[0][:frame], ucardata_y[0][:frame])
        ln4.set_data(ucardata_x[1][:frame], ucardata_y[1][:frame])

        ln5.set_data(eddata_x[:frame], eddata_y[:frame])
        # text_pt.set_text("reward =%.3f,\n epsiode reward = %.3f" % (reward[frame],sum(reward[:frame])))
        # text_pt.set_text("uav0_z =%.3f,\n uav1_z = %.3f \n reward =%.3f,\n epsiode reward = %.3f" % (trajectroy["Uavz"][0][frame],trajectroy["Uavz"][1][frame],reward[frame],sum(reward[:frame])))
        text_pt.set_text("uav0_z =%.3f,action0 =%.3f,\n uav1_z = %.3f , action1 = %.3f" % (
        trajectroy["Uavz"][0][frame], trajectroy["action"][0][frame],trajectroy["Uavz"][1][frame], trajectroy["action"][1][frame]))
        return ln1,ln2,ln3,ln4,ln5,text_pt

    '''
    函数FuncAnimation(fig,func,frames,init_func,interval,blit)是绘制动图的主要函数，其参数如下：
        a.fig 绘制动图的画布名称
        b.func自定义动画函数，即下边程序定义的函数update
        c.frames动画长度，一次循环包含的帧数，在函数运行时，其值会传递给函数update(n)的形参“n”
        d.init_func自定义开始帧，即传入刚定义的函数init,初始化函数
        e.interval更新频率，以ms计
        f.blit选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显
    '''

    ani = FuncAnimation(fig=fig, func=update, frames=range(len(trajectroy["Uavx"][0])),interval=100,
                        init_func=init, blit=True)
    plt.legend(("uav0","uav1", "ucar1", "ucar2","ED"), loc="upper right")
    plt.show()
    if f==None:
        ani.save('trajectory.gif')
    else:
        ani.save(f)
    plt.close()
    return 0

    # ani.save('sin_test1.gif', writer='imagemagick', fps=100)

