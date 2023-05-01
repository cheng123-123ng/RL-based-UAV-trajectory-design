from two_agent_dependent import two_agent_real_dependent
from two_agent_com import two_agent_com
import matplotlib.pyplot as plt
max_frames =2000


import numpy as np
#####
rewards_com = two_agent_com(max_frames)

rewards_dep = two_agent_real_dependent(max_frames)
plt.figure()
mid = np.arange(max_frames)
plt.plot( mid,rewards_com,'r',mid,rewards_dep, 'b')
plt.legend(['com','dep'])
plt.show()

