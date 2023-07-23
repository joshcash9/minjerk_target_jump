import numpy as np
import os, math, operator, random, csv, scipy
import csv
from scipy.optimize import curve_fit
from scipy import special
from scipy import stats
from scipy.optimize import minimize
from pylab import *

############################### ############################### #############################
################################ 1D minjerk with target jumps ###############################
############################### ############################### #############################

h, s = 0.0001, 1.0 # step size (s), simulation time (s)
steps = int(s/h) + 1 

# Initial conditions
target, x, xdot, xddot = 10.0, 0.0, 0.0, 0.0
D = target - x
q0 = np.array([[x],[xdot],[xddot]])

# Matrices to store states
Q = np.zeros((steps, 3, 1))
Q[0] = q0

for n in range(steps-1):
    D = s - (n)*h
    print(D)
    A = np.array([[0,1,0],[0,0,1],[-60.0/D**3.0,-36.0/D**2,-9.0/D]])
    B = np.array([[0],[0],[60.0/D**3]])
    C = np.identity(3)
    Ad = A * h + C # trick to define ODE and update
    Bd = B * h
    if n > 5000:
        target = 15.0
    Q[n+1] = np.matmul(Ad, Q[n]) + Bd * target


Q0 = np.reshape(Q, (steps, np.shape(q0)[0]))
Q1 = np.transpose(Q0)

TIME1 = np.linspace(0,(steps-1) * h,steps)
# PLOT STATES OVER TIME
axiscolor = 'k'
titlefsize = 32
axisfsize = 28
scalefsize = 24 
fig, ax = plt.subplots(figsize=(8,8))
#axis([0,s,-2.0,4])
plt.title('Target Jumps', fontsize = titlefsize)
plt.xlabel('Time (s)', fontsize = scalefsize)
plt.ylabel('Position (m) and Velocity (m/s)',fontsize = scalefsize)
d1 = errorbar(TIME1,Q1[0], linestyle = '-', linewidth = 2.0, color = '#FD8B0B')
d2 = errorbar(TIME1,Q1[1], linestyle = '-', linewidth = 2.0, color = '#0BB8FD')
#d3 = errorbar(TIME1,Q1[2], linestyle = '-', linewidth = 2.0, color = '0.5')
lg = legend([d1,d2], ['Position','Velocity','Acceleration'], fontsize = 18, loc = 'upper left', ncol = 1)
lg.get_frame().set_linewidth(0.0)
lg.get_frame().set_alpha(0.0)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=False, direction='in')
#plt.xticks(np.arange(0, n*h, 0.25))
#plt.yticks(np.arange(-1.0, 5, 1.0))
ax.spines['left'].set_color(axiscolor)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(axiscolor)
ax.tick_params(axis='x', colors=axiscolor, labelsize = scalefsize)
ax.tick_params(axis='y', colors=axiscolor, labelsize = scalefsize)
plt.tight_layout()
show()

############################### ############################### #############################
################################ 2D minjerk with target jumps ###############################
############################### ############################### #############################

h, s = 0.0001, 1.0 # step size (s), simulation time (s)
steps = int(s/h) 

# Initial conditions
target_x, x, xdot, xddot, target_y, y, ydot, yddot = 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0
q0 = np.array([[x],[xdot],[xddot],[y],[ydot],[yddot]])
# Matrices to store states
Q = np.zeros((steps, 6, 1))
Q[0] = q0
for n in range(steps-1):
    D = s - (n)*h
    print(D)
    if n > 5000:
        target_x, target_y = 5.0, 10.0
    A = np.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[-60.0/D**3.0,-36.0/D**2,-9.0/D,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,-60.0/D**3.0,-36.0/D**2,-9.0/D]])
    B = np.array([[0],[0],[(60.0/D**3)*target_x],[0],[0],[(60.0/D**3)*target_y]])
    C = np.identity(6)
    Ad = A * h + C # trick to define ODE and update
    Bd = B * h 
    Q[n+1] = np.matmul(Ad, Q[n]) + Bd


Q0 = np.reshape(Q, (steps, np.shape(q0)[0]))
Q1 = np.transpose(Q0)

TIME1 = np.linspace(0,(steps-1) * h,steps)
# PLOT STATES OVER TIME
axiscolor = 'k'
titlefsize = 32
axisfsize = 28
scalefsize = 24 
fig, ax = plt.subplots(figsize=(8,8))
#axis([0,s,-2.0,4])
plt.title('Target Jumps - X coordinates', fontsize = titlefsize)
plt.xlabel('Time (s)', fontsize = scalefsize)
plt.ylabel('Position (m) and Velocity (m/s)',fontsize = scalefsize)
d1 = errorbar(TIME1,Q1[0], linestyle = '-', linewidth = 2.0, color = '#FD8B0B')
d2 = errorbar(TIME1,Q1[1], linestyle = '-', linewidth = 2.0, color = '#0BB8FD')
d3 = errorbar(TIME1,Q1[2], linestyle = '-', linewidth = 2.0, color = '0.5')
lg = legend([d1,d2,d3], ['Position','Velocity','Acceleration'], fontsize = 18, loc = 'upper left', ncol = 1)
lg.get_frame().set_linewidth(0.0)
lg.get_frame().set_alpha(0.0)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=False, direction='in')
#plt.xticks(np.arange(0, n*h, 0.25))
#plt.yticks(np.arange(-1.0, 5, 1.0))
ax.spines['left'].set_color(axiscolor)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(axiscolor)
ax.tick_params(axis='x', colors=axiscolor, labelsize = scalefsize)
ax.tick_params(axis='y', colors=axiscolor, labelsize = scalefsize)
plt.tight_layout()
show()

# PLOT STATES OVER TIME
axiscolor = 'k'
titlefsize = 32
axisfsize = 28
scalefsize = 24 
fig, ax = plt.subplots(figsize=(8,8))
#axis([0,s,-2.0,4])
plt.title('Target Jumps - Y coordinates', fontsize = titlefsize)
plt.xlabel('Time (s)', fontsize = scalefsize)
plt.ylabel('Position (m) and Velocity (m/s)',fontsize = scalefsize)
d1 = errorbar(TIME1,Q1[3], linestyle = '-', linewidth = 2.0, color = '#FD8B0B')
d2 = errorbar(TIME1,Q1[4], linestyle = '-', linewidth = 2.0, color = '#0BB8FD')
d3 = errorbar(TIME1,Q1[5], linestyle = '-', linewidth = 2.0, color = '0.5')
lg = legend([d1,d2,d3], ['Position','Velocity','Acceleration'], fontsize = 18, loc = 'upper left', ncol = 1)
lg.get_frame().set_linewidth(0.0)
lg.get_frame().set_alpha(0.0)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=False, direction='in')
#plt.xticks(np.arange(0, n*h, 0.25))
#plt.yticks(np.arange(-1.0, 5, 1.0))
ax.spines['left'].set_color(axiscolor)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(axiscolor)
ax.tick_params(axis='x', colors=axiscolor, labelsize = scalefsize)
ax.tick_params(axis='y', colors=axiscolor, labelsize = scalefsize)
plt.tight_layout()
show()

# 2D HAND TRAJECTORIES
xaxis = np.linspace(0, (s + targetpad), len(Xlqg1[0]))

axiscolor = 'k'
fsize = 'auto'
titlefsize = 27
axisfsize = 24
scalefsize = 21 
fig, ax = plt.subplots(figsize=(9,8))
#plt.title('Grip Force Rate', color=axiscolor, fontsize = titlefsize)
plt.xlabel('X position', color = axiscolor, fontsize = axisfsize)
plt.ylabel('Y Position', color = axiscolor, fontsize = axisfsize)
axis([-6.0, 6.0, -1.0, 11.0])
#plot([50.0, 50.0], [-1.0, 1.0], color = '0.75', linestyle='--', linewidth=1.0)
#plot([170.0, 170.0], [-1.0, 1], color = '0.75', linestyle='--', linewidth=1.0)
#plot([185.0, 185.0], [-1.0, 1], color = '0.75', linestyle='--', linewidth=1.0)
#ax.text(10, 0.8, 'Baseline', color = '0.75', fontsize = axisfsize)

#d1 = scatter(Xlqg1[0], Xlqg1[1], color = '#FD8B0B', linestyle='-', linewidth = 0.5)
d1 = errorbar(Q1[0], Q1[3], color = '#FD8B0B', linestyle='-', linewidth = 2.0)
lg = legend([d1], ['Pos'], loc='lower center', fontsize = axisfsize, ncol=2)
lg.get_frame().set_linewidth(0.0)
lg.get_frame().set_alpha(0.0)
# next 7 lines chaange legend colours
ltext = plt.gca().get_legend().get_texts()
plt.setp(ltext[0], color='#FD8B0B')
plt.setp(ltext[1], color='#FD8B0B')
plt.setp(ltext[2], color='#0BB8FD')
plt.setp(ltext[3], color='#0BB8FD')
#plt.setp(ltext[4], color='0.5')
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=False, top=False, left=False, right=False)
#plt.yticks(np.arange(0, 20.0, 2.5))
ax.spines['left'].set_color(axiscolor)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(axiscolor)
ax.tick_params(axis='x', colors=axiscolor, labelsize = scalefsize)
ax.tick_params(axis='y', colors=axiscolor, labelsize = scalefsize)
plt.tight_layout()
#fig.savefig('/Users/joshcash/Desktop/parabola' + '.pdf', transparent=True)
show()



