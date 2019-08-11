import obspy
from obspy import core
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
import re
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import random
import time
from numpy import matrix as mat


##-------------------Advanced Holcomb's method-------------------
def LV(BHE,BHN,BLE,BLN):  
    s_time = time.clock()
    coff_e,coff_n = [],[]
    scoff_e,scoff_n = [],[] 
    for r in np.linspace(0,2*np.pi,72):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        coff_e.append(np.corrcoef(new_bhe.data,BLE.data)[0][1])
    for r in np.linspace((np.argmax(coff_e)*5-5)/180*np.pi,(np.argmax(coff_e)*5+5)/180*np.pi,100):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        scoff_e.append(np.corrcoef(new_bhe.data,BLE.data)[0][1])
    maxc_e=np.max(scoff_e)
    angle_e = np.argmax(coff_e)*5 -5 + np.argmax(scoff_e)*0.1
    for r in np.linspace(0,2*np.pi,72):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        coff_n.append(np.corrcoef(new_bhn.data,BLN.data)[0][1])
    for r in np.linspace((np.argmax(coff_e)*5-5)/180*np.pi,(np.argmax(coff_n)*5+5)/180*np.pi,100):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        scoff_n.append(np.corrcoef(new_bhn.data,BLN.data)[0][1])
    maxc_n=np.max(scoff_n)
    angle_n = np.argmax(coff_n)*5 -5 + np.argmax(scoff_n)*0.1
    
    W = np.mat([[maxc_e],[maxc_n]])/(maxc_e+maxc_n)  # weight 
    angle = float([angle_e,angle_n]*W)
    e_time = time.clock()
    timecost = e_time-s_time
    plt.plot(np.linspace(0,360,72), coff_e)
    plt.plot(np.linspace(0,360,72), coff_n)
    plt.title("Correlation coefficient curves")
    plt.legend(("E-W component","N-S component"))
    plt.plot(angle,[maxc_e,maxc_n]*W,"x",)  # peak 
    plt.annotate("("+str(round(angle,2))+","+str(round((maxc_n+maxc_e)/2,3))+")", xy = (angle, 0.8),)
    plt.grid(True)
    plt.xlabel("Angle(°)")
    plt.ylabel("Correlation coefficient")
    plt.show()
    print("time cost:%0.3f"%(timecost),"angle:%f"%(angle))
    return angle/180*np.pi



##-------------------Ringler's method-------------------

#  Cost function
def Func(abc,iput):
    a = abc[0,0]
    b = abc[1,0]
    c = abc[2,0]
    return np.corrcoef((correction(iput[:,0],iput[:,1],a)[0].data[2000:]),iput[2000:,2])[0,1]

#  Advanced cost function with damping term
def Func_s(abc,iput,corrm,a1):
    a = abc[0,0]
    b = abc[1,0]
    c = abc[2,0]
    return np.corrcoef((correction(iput[:,0],iput[:,1],a)[0].data[2000:]),iput[2000:,2])[0,1]+((corrm-1)*(a-a1))**2

# Calculate derivative   
def Deriv(abc,iput,n):
    x1 = abc.copy()
    x2 = abc.copy()
    x1[n,0] -= 0.000001
    x2[n,0] += 0.000001
    p1 = Func(x1,iput)
    p2 = Func(x2,iput)
    d = (p2-p1)*1.0/(0.000002)
    return d

# Use Levenberg–Marquardt algorithm to calculate
# data length: 2000s Xr=BLE.data[:2000*50]*4,Xt=BHE.data[:2000*50],Yt=BHN.data[:2000*50],
def R(BHE,BHN,BLE,BLN):
    Xr=BLE.data[:2000*50]*4
    Xt=BHE.data[:2000*50]
    Yt=BHN.data[:2000*50]
    hh = np.vstack((np.array(Xt),np.array(Yt),np.array(Xr))).T
    h = hh
    # h = [0]*(int(len(hh)/500)-1)
    # h = h.reshape((4,25000,3))
    # for n in range(len(h)):
        # h[n] = hh[n*500:n*500+2000]
    # n = len(h)
    n = 1
    y = np.ones(n)
    y = mat(1)
    J = mat(np.zeros((n,3)))      #雅克比矩阵
    fx = mat(np.zeros((n,1)))     # f(x)  100*1  误差
    fx_tmp = mat(np.zeros((n,1)))
    xk = mat([[0.0],[0.0],[0.0]]) # 参数初始化
    lase_mse = 0
    step = 0
    u,v= 1,2
    conve = 100
    s_time = time.clock()
    while (conve):
        mse,mse_tmp = 0,0
        step += 1  
        for i in range(n):
            fx[i] =  Func(xk,h) - y[i]    # 注意不能写成  y - Func  ,否则发散
            mse += fx[i]**2
            for j in range(3): 
                J[i,j] = Deriv(xk,h,j)
                
        mse /= n
        H = J.T*J + np.eye(3)*float(u)   # 3*3
        dx = -H.I * J.T*fx        # 注意这里有一个负号，和fx = Func - y的符号要对应
        xk_tmp = xk.copy()
        xk_tmp += dx
        for j in range(n):
            fx_tmp[i] =  Func(xk_tmp,h) - y[i]  
            mse_tmp += fx_tmp[i]**2
            
        mse_tmp /= n
        q = (mse - mse_tmp)/((0.5*dx.T*(float(u)*dx - J.T*fx))[0,0])
        if q > 0:
            s = 1.0/3.0
            v = 2
            mse = mse_tmp
            xk = xk_tmp
            temp = 1 - pow(2*q-1,3)
            if s > temp:
                u = u*s
            else:
                u = u*temp
        else:
            u = u*v
            v = 2*v
            xk = xk_tmp
        # print("step = %d,abs(mse-lase_mse) = %.8f" %(step,np.abs(mse-lase_mse)))  
        if abs(mse-lase_mse)<0.000001:
            # e_time = time.clock()
            break
           
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
    # print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(xk[0,0]/np.pi*180))
    conve = 100
    corrm = Func(xk,h)
    a1=xk[0,0]
    while (conve):
        mse,mse_tmp = 0,0
        step += 1  
        for i in range(n):
            fx[i] =  Func_s(xk,h,corrm,a1) - y[i]    # 注意不能写成  y - Func  ,否则发散
            mse += fx[i]**2
            for j in range(3): 
                J[i,j] = Deriv(xk,h,j)
                
        mse /= n
        H = J.T*J + np.eye(3)*float(u)   # 3*3
        dx = -H.I * J.T*fx        # 注意这里有一个负号，和fx = Func - y的符号要对应
        xk_tmp = xk.copy()
        xk_tmp += dx
        for j in range(n):
            fx_tmp[i] =  Func_s(xk_tmp,h,corrm,a1) - y[i]  
            mse_tmp += fx_tmp[i]**2
            
        mse_tmp /= n
        q = (mse - mse_tmp)/((0.5*dx.T*(float(u)*dx - J.T*fx))[0,0])
        if q > 0:
            s = 1.0/3.0
            v = 2
            mse = mse_tmp
            xk = xk_tmp
            temp = 1 - pow(2*q-1,3)
            if s > temp:
                u = u*s
            else:
                u = u*temp
        else:
            u = u*v
            v = 2*v
            xk = xk_tmp
        # print("step = %d,abs(mse-lase_mse) = %.8f" %(step,np.abs(mse-lase_mse)))  
        if abs(mse-lase_mse)<(1/10e14):
            e_time = time.clock()
            break
           
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
    angle = float(xk[0,0])
    if angle<0:
        angle += 2*np.pi
    print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(angle/np.pi*180),"step = %d"%step)
    return angle


##-------------------Our method-------------------
def MY(BHE,BHN,BLE,BLN):
    s_time=time.clock()
    ca,sa = calculate_parameters(BHE,BHN,BLE,BLN)
    rec_ca,rec_sa = r_rebuild(ca,sa)
    angle = getangle(rec_ca,rec_sa)
    e_time = time.clock()
    print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(angle/np.pi*180))
    return angle
	

##-------------------Comparison of three method-------------------
angle = LV(BHE,BHN,BLE,BLN)
new_bhe,new_bhn = correction(BHE, BHN ,angle)
angle_1 = R(BHE,BHN,BLE,BLN)
new_bhe_1,new_bhn_1 = correction(BHE, BHN ,angle_1)
angle_2 = MY(BHE,BHN,BLE,BLN)
new_bhe_2,new_bhn_2 = correction(BHE, BHN ,angle_2)

fig,axs = plt.subplots(4,1)
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[3].grid(True)
axs[0].set_ylabel("Counts")
axs[1].set_ylabel("Counts")
axs[2].set_ylabel("Counts")
axs[3].set_ylabel("Counts")
axs[0].plot((new_bhe.data[-500:]-np.mean(new_bhe.data[-500:])),color = 'b')
axs[1].plot((new_bhe_1.data[-500:]-np.mean(new_bhe_1.data[-500:])),color = 'b')
axs[2].plot((new_bhe_2.data[-500:]-np.mean(new_bhe_2.data[-500:])),color = 'b')
axs[3].plot(((BLE.data[-500:]-np.mean(BLE.data[-500:]))*4),color = 'g')
axs[0].legend(["Correcred BHE data by Holcomb's method"],loc="lower left")
axs[1].legend(["Correcred BHE data by Ringler's method"],loc="lower left")
axs[2].legend(["Correcred BHE data by proposed method"],loc="lower left")
axs[3].legend(["BLE data "],loc="lower left")

fig,axs = plt.subplots(4,1)
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[3].grid(True)
axs[0].set_ylabel("Counts")
axs[1].set_ylabel("Counts")
axs[2].set_ylabel("Counts")
axs[3].set_ylabel("Counts")
axs[0].plot((new_bhn.data[-500:]-np.mean(new_bhn.data[-500:])),color = 'b')
axs[1].plot((new_bhn_1.data[-500:]-np.mean(new_bhn_1.data[-500:])),color = 'b')
axs[2].plot((new_bhn_2.data[-500:]-np.mean(new_bhn_2.data[-500:])),color = 'b')
axs[3].plot(((BLN.data[-500:]-np.mean(BLN.data[-500:]))*4),color = 'g')
axs[0].legend([r"Correcred BHN data by Holcomb's method"],loc="lower left")
axs[1].legend([r"Correcred BHN data by Ringler's method"],loc="lower left")
axs[2].legend([r"Correcred BHN data by proposed method"],loc="lower left")
axs[3].legend(["BLN data"],loc="lower left")
plt.show()

