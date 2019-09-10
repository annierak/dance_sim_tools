import numpy as np
import matplotlib.pyplot as plt
import dance_sim_tools.trace_objects as trace_objects
# import odor_tracking_sim.utility as utility


#Test the ird trace by time

num_segments = 20

ks = np.random.normal(1,0.2,num_segments)

ird0 = np.radians(70)
irds = np.zeros_like(ks)
irds[0] = ird0

for i in range(1,len(ks)):
    irds[i] = ks[i-1]*irds[i-1]

r_0 = np.radians(50)
velocity = np.radians(3)
food1_loc = np.radians(-10)


trace = trace_objects.DistanceListTrace(
        food1_loc,irds,r_0,t_stop=2,velocity=velocity)

print(np.unique(np.diff(trace.path)/(np.diff(trace.t))))

plt.figure()
plt.plot(trace.t,trace.path)
plt.show()
