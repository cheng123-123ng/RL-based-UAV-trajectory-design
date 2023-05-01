# 动态画出sin函数曲线
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
def render(trajectroy=None, num=0 ,f = None):

    fig, ax = plt.subplots()
    ax.grid()
    xdata, ydata = [], []
    xdata1, ydata1 = [], []
    xdata2, ydata2 = [], []
    """"
    trajectory = {
        "Uavx":[],"Uavy":[],
        "Ucar1x":[],"Ucar1y":[],
        "Ucar2x":[],"Ucar2y":[],
        "reward":[],
    }
    """
    if trajectroy != None:
        xdata = trajectroy["Uavx"]
        ydata = trajectroy["Uavy"]
        xdata1 = trajectroy["Ucar1x"]
        ydata1 = trajectroy["Ucar1y"]
        xdata2 = trajectroy["Ucar2x"]
        ydata2 = trajectroy["Ucar2y"]
        reward = trajectroy["reward"]
    ln1, = ax.plot([], [], 'bo')
    ln2, = ax.plot([], [], 'g+-')
    ln3, = ax.plot([], [], 'r-')

    text_pt = plt.text(4, 0.8, '', fontsize=16)

    def init():
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 21)
        # 返回曲线
        return ln1,ln2,ln3


    def update(frame):
        # 将每次传过来的n追加到xdata中
        # 重新设置曲线的值
        ln1.set_data(xdata[:frame], ydata[:frame])

        # 重新设置曲线的值
        ln2.set_data(xdata1[:frame], ydata1[:frame])

        ln3.set_data(xdata2[:frame], ydata2[:frame])
        text_pt.set_text("reward =%.3f,\n epsiode reward = %.3f" % (reward[frame],sum(reward[:frame])))
        return ln1,ln2,ln3,text_pt

    '''
    函数FuncAnimation(fig,func,frames,init_func,interval,blit)是绘制动图的主要函数，其参数如下：
        a.fig 绘制动图的画布名称
        b.func自定义动画函数，即下边程序定义的函数update
        c.frames动画长度，一次循环包含的帧数，在函数运行时，其值会传递给函数update(n)的形参“n”
        d.init_func自定义开始帧，即传入刚定义的函数init,初始化函数
        e.interval更新频率，以ms计
        f.blit选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显
    '''

    ani = FuncAnimation(fig=fig, func=update, frames=range(len(trajectroy["Uavx"])),interval=100,
                        init_func=init, blit=True)
    plt.legend(("uav","ucar1","ucar2"),loc="upper right")
    plt.show()
    if f==None:
        ani.save('trajectory.gif')
    else:
        ani.save(f)
    plt.close()
    return 0

    # ani.save('sin_test1.gif', writer='imagemagick', fps=100)

