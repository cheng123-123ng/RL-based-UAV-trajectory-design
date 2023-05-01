# 动态画出sin函数曲线
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
def render(trajectroy=None,reward=None,f = None):

    fig, ax = plt.subplots()
    ax.grid()
    xdata, ydata = trajectroy[0][0], trajectroy[0][1]
    xdata1, ydata1 = trajectroy[1][0], trajectroy[1][1]

    xdata2, ydata2 = trajectroy[2][0], trajectroy[3][1]
    xdata3, ydata3 = trajectroy[3][0], trajectroy[4][1]
    xdata4, ydata4 = trajectroy[4][0], trajectroy[5][1]
    xdata5, ydata5 = trajectroy[5][0], trajectroy[6][1]
    xdata6, ydata6 = trajectroy[6][0], trajectroy[7][1]

    ln1, = ax.plot([], [], 'b+')
    ln2, = ax.plot([], [], 'g+')
    ln3, = ax.plot([], [], 'ko')
    ln4, = ax.plot([], [], 'ko')
    ln5, = ax.plot([], [], 'ko')
    ln6, = ax.plot([], [], 'ko')
    ln7, = ax.plot([], [], 'ko')

    text_pt = plt.text(4, 0.8, '', fontsize=16)

    def init():
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 21)
        # 返回曲线
        return ln1,ln2,ln3,ln4


    def update(frame):
        ln1.set_data(xdata[:frame], ydata[:frame])
        ln2.set_data(xdata1[:frame], ydata1[:frame])

        ln3.set_data(xdata2[:frame], ydata2[:frame])
        ln4.set_data(xdata3[:frame], ydata3[:frame])
        ln5.set_data(xdata4[:frame], ydata4[:frame])
        ln6.set_data(xdata5[:frame], ydata5[:frame])
        ln7.set_data(xdata6[:frame], ydata6[:frame])

        text_pt.set_text("reward =%.3f,\n epsiode reward = %.3f" % (reward[frame],sum(reward[:frame])))
        return ln1,ln2,ln3,ln4,text_pt

    '''
    函数FuncAnimation(fig,func,frames,init_func,interval,blit)是绘制动图的主要函数，其参数如下：
        a.fig 绘制动图的画布名称
        b.func自定义动画函数，即下边程序定义的函数update
        c.frames动画长度，一次循环包含的帧数，在函数运行时，其值会传递给函数update(n)的形参“n”
        d.init_func自定义开始帧，即传入刚定义的函数init,初始化函数
        e.interval更新频率，以ms计
        f.blit选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显
    '''

    ani = FuncAnimation(fig=fig, func=update, frames=range(len(trajectroy[0][0])),interval=100,
                        init_func=init, blit=True)
    plt.legend(("uav", "ucar1", "ucar2","ED"), loc="upper right")
    plt.show()
    if f==None:
        ani.save('trajectory.gif')
    else:
        ani.save(f)
    plt.close()
    return 0

    # ani.save('sin_test1.gif', writer='imagemagick', fps=100)

