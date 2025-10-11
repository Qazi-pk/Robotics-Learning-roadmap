import matplotlib.pyplot as plt
import numpy as np

def plot_joint_trajectories(t, q_sol, labels=('q1','q2')):
    plt.figure(figsize=(8,4))
    for i in range(q_sol.shape[0]):
        plt.plot(t, q_sol[i,:], label=labels[i])
    plt.legend(); plt.grid(True); plt.xlabel('Time [s]'); plt.ylabel('Angle [rad]')

def plot_end_effector_path(qs, L1, L2, title='End-effector path'):
    x = L1*np.cos(qs[0,:]) + L2*np.cos(qs[0,:]+qs[1,:])
    y = L1*np.sin(qs[0,:]) + L2*np.sin(qs[0,:]+qs[1,:])
    plt.figure(figsize=(5,5))
    plt.plot(x,y); plt.scatter([x[0], x[-1]],[y[0], y[-1]], c=['green','red'])
    plt.title(title); plt.axis('equal'); plt.grid(True)
