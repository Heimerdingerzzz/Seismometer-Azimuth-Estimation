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

# Reading seismograms file, selecting traces in a certain frequency and show the waveform.
# You can save the waveform to a figure file in some formats.
def readfile():
    print("please select file(s):")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames()
    root.destroy()
    datastream = obspy.core.stream.Stream()
    for dir in file_path:
        print(dir)
        datastream += obspy.read(dir)
    samplerate = float(input("sample rate："))
    for tr in datastream:
        if tr.stats.sampling_rate != samplerate:
            datastream.remove(tr)
    for tr in datastream.select(component = "Z"):
        datastream.remove(tr)
    datastream.merge()
    outfile = input("output file (support png, pdf, ps, eps and svg):\n")
    datastream.plot(dpi=300,outfile=outfile)
    # mypath: "C:\Users\ironman\Desktop\idea\论文\使用小波分解重构法计算地震计安装方位角\datastream.svg"
    print(datastream,"\n")
    return datastream

# Do some pretreatment such as choosing the time period and calculating both Ca and Sa parameters.
def process(Stream):
    namehead = re.search(r"QT\.\d+\..",Stream[0].id)[0]
    global BHE,BHN,BLE,BLN
    BHE = Stream.select(id = namehead+"BHE")[0]
    BHN = Stream.select(id = namehead+"BHN")[0]
    BLE = Stream.select(id = namehead+"BLE")[0]
    BLN = Stream.select(id = namehead+"BLN")[0]
    
    start_time_default = UTCDateTime(np.max([BHE.stats.starttime.timestamp,\
                                            BHN.stats.starttime.timestamp,\
                                            BLE.stats.starttime.timestamp,\
                                            BLN.stats.starttime.timestamp]))
    end_time_default = UTCDateTime(np.min([BHE.stats.endtime.timestamp,\
                                            BHN.stats.endtime.timestamp,\
                                            BLE.stats.endtime.timestamp,\
                                            BLN.stats.endtime.timestamp]))
    while(True):
        key = input("use default time ?(Y/N)\n")
        if key=="Y" or key=="y":
            startt = start_time_default
            endt = end_time_default
            break
        elif key=="N" or key=="n":
            startt = core.UTCDateTime(input("input start time:"))
            if startt.timestamp<start_time_default.timestamp or startt.timestamp >end_time_default.timestamp:
                startt = start_time_default
                print("warning: start time out of range. replaced with default start time")
            endt = core.UTCDateTime(input("input end time:"))
            if endt.timestamp > end_time_default.timestamp or endt.timestamp<start_time_default.timestamp:
                endt = end_time_default
                print("warning: end time out of range. replaced with default end time")
            if startt.timestamp > endt.timestamp:
                startt = start_time_default
                endt = end_time_default
                print("warning: wrong input! replaced with default time")
            break
        else :
            print("input error,try again")
    BHE.trim(startt,endt)
    BHN.trim(startt,endt)
    BLE.trim(startt,endt)
    BLN.trim(startt,endt)
    
    return BHE,BHN,BLE,BLN

# Calculate Ca and Sa parameters
def calculate_parameters(BHE,BHN,BLE,BLN):
    x1 = BHE.data-np.mean(BHE.data)
    y1 = BHN.data-np.mean(BHN.data)
    x2 = (BLE.data-np.mean(BLE.data))*4
    y2 = (BLN.data-np.mean(BLN.data))*4

    ca = (x1*x2+y1*y2)/(x2**2+y2**2)
    sa = (x1*y2-x2*y1)/(x2**2+y2**2)

    return ca,sa

# Pauta method to exclude exception data
def pauta(t):
    flag = 1
    while(flag==1):
        std = np.std(t, ddof = 1)
        aver = np.mean(t)
        # print(std,aver)
        for n in t:
            if (abs(n - aver)) > 3*std:
                # print(n,t.index(n))
                t.remove(n)
                flag = 1
                break
            else:
                flag = 0
    return t 

# Sigmoid normlized
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

# Wavelet filter for Ca Sa.  
def r_rebuild(ca, sa, level=None):
    max_level = pywt.dwt_max_level(len(ca),2)  #calculate max level
    if level == None:
        level = max_level
    elif level>max_level :
        level = max_level
        print("Decomposition level is out of range! Setting to default.")
    ## process Ca
    coeffs = pywt.wavedec(ca, 'db1', level=level)  # decomposition by haar wavelet
    usecoff = [coeffs[0]]  # retain the approximation
    for n in range(len(coeffs)-1):
        usecoff.append(np.zeros(len(coeffs[n+1]))) # set detial components to zero 
    rec_ca = pywt.waverec(usecoff, 'db1')   # reconstruction
    ## process Sa
    coeffs = pywt.wavedec(sa, 'db1', )
    usecoff = [coeffs[0]]
    for n in range(len(coeffs)-1):
        usecoff.append(np.zeros(len(coeffs[n+1])))
    rec_sa = pywt.waverec(usecoff, 'db1')
    return rec_ca,rec_sa

# Estimate the azimuth
def getangle(ca,sa):
    # calculate the reference value of azimuth
    cos_alpha = np.arccos(np.mean(random.sample(list(ca),10)))
    sin_alpha = np.arcsin(np.mean(random.sample(list(sa),10)))
    #print(cos_alpha,sin_alpha)
    
    # determine the quadrant that the azimuth is belong to
    if 0 < cos_alpha < np.pi/2 and 0 < sin_alpha < np.pi/2 :
        print('转动角在第一象限')
        alpha = (cos_alpha+sin_alpha)/2
    elif np.pi/2 < cos_alpha < np.pi and 0 < sin_alpha < np.pi/2 :
        print('转动角在第二象限')
        alpha = (cos_alpha+np.pi-sin_alpha)/2
    elif np.pi/2 < cos_alpha < np.pi and -np.pi/2 < sin_alpha < 0:
        print('转动角在第三象限')
        alpha = (2*np.pi-cos_alpha+np.pi-sin_alpha)/2
    elif 0 < cos_alpha < np.pi/2 and -np.pi/2 < sin_alpha < 0:
        print('转动角在第四象限')
        alpha = (2*np.pi-cos_alpha+2*np.pi+sin_alpha)/2
    else :
        print("error occur")
        return 0
    print(alpha*180/np.pi)
    return alpha

# Correct the raw data
def correction(h_e, h_n ,angle):
    h_e_new = h_e.copy()
    h_n_new = h_n.copy()
    h_e_new.data = h_e.data*np.cos(angle) + h_n.data*np.cos(angle + np.pi/2) 
    h_n_new.data = h_e.data*np.sin(angle) + h_n.data*np.sin(angle + np.pi/2)
    return h_e_new,h_n_new

# Show the histogram of "Ca Sa" parameters 
def plothist(ca,sa):
    ca_mu = np.mean(ca)
    ca_sigma = np.std(ca, ddof =1)
    sa_mu = np.mean(sa)
    sa_sigma = np.std(sa, ddof =1)
    
    fig,ax = plt.subplots(2,1,dpi=300)
    n, bins, patches = ax[0].hist(ca, 1000, range=(ca_mu-3*ca_sigma,ca_mu+3*ca_sigma),color = 'g',alpha=0.75)
    ax[0].set_title('Parameter Ca')
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Frequency")

    n, bins, patches = ax[1].hist(sa, 1000, range=(sa_mu-3*sa_sigma,sa_mu+3*sa_sigma),alpha=0.75)
    ax[1].set_title('Parameter Sa')
    ax[1].set_xlabel("Value", )
    ax[1].set_ylabel("Frequency", )
    plt.subplots_adjust(hspace=0.5)
    #plt.savefig(r"/home/anotherone/文档/mycode/Ca&Sa.svg",format="svg")
    # plt.savefig(r"C:\Users\ironman\Desktop\idea\论文\使用小波分解重构法计算地震计安装方位角\英文投稿\figures\Ca&Sa.jpg",format="jpg")
    plt.tight_layout()
    plt.show()

# Illustrate the effect of proposed method by different level
def illustration(ca,sa, level= None):
    max_level = pywt.dwt_max_level(len(ca),2)
    
    fig = plt.figure("illustration",dpi=300)
    
    if level==None:
        level = max_level
        # plt.suptitle("Decomposition level: max",size = 16, color= "blue")
        # plt.text(ca_mu-4*ca_sigma, np.max(n_1), "wavedeposition level: max",size = 5, color= "blue",\
        # bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
    elif level>max_level :
        level = max_level
        print("Decomposition level is out of range! Setting to default.")
        # plt.suptitle("Decomposition level: "+str(level),size = 16, color= "blue")
    else:
        pass
        # plt.suptitle("Decomposition level: "+str(level),size = 16, color= "blue")
        # plt.text(ca_mu-4*ca_sigma, np.max(n_1), "wavedeposition level: "+str(level), size = 5, color= "blue",\
        # bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
    rec_ca,rec_sa = r_rebuild(ca, sa, level=level)
    
    ca_mu = np.mean(ca)
    ca_sigma = np.std(ca, ddof =1)
    sa_mu = np.mean(sa)
    sa_sigma = np.std(sa, ddof =1)
    
    ax1 = fig.add_subplot(1,2,1)
    n_1, bins_1, patches_1 = ax1.hist(ca,bins = 1000,range=(ca_mu-6*ca_sigma,ca_mu+6*ca_sigma),)
    ##   plt.ylabel("Log",fontsize=16)
    
    rec_ca_mu = np.mean(rec_ca)
    rec_ca_sigma = np.std(rec_ca, ddof =1)
    rec_sa_mu = np.mean(rec_sa)
    rec_sa_sigma = np.std(rec_sa, ddof =1)
    n_2, bins_2, patches_2 = ax1.hist(rec_ca,bins = 1000,range=(ca_mu-6*ca_sigma,ca_mu+6*ca_sigma),alpha=0.75,)
    ax1.legend(("Original histogram","Processed histogram "),loc="upper right")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(ca,linewidth=0.8)
    ax2.plot(rec_ca,linewidth=0.8)
    ax2.legend(("Before processing","After processing"),loc="upper right")
    plt.xticks(fontsize=8,rotation=-30)
    plt.yticks(fontsize=8)
    ##    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.05,wspace=0.1)
    plt.tight_layout()
    plt.show()

# Show the comparison 
def effect(BHE,BHN,BLE,BLN,new_bhe,new_bhn):
    # Note: select 1000 points for illustrate the detial
    fig = plt.figure("Raw data of E-W component",dpi=300)
    axs = fig.subplots()
    axs.set_title("Raw BHE and Raw BLE data")
    axs.set_ylabel("Counts")
    axs.grid(True)
    axs.plot((BHE.data[-1000:]-np.mean(BHE.data[-1000:])),linewidth=0.8)
    axs.plot(((BLE.data[-1000:]-np.mean(BLE.data[-1000:]))*4),linewidth=0.8)
    axs.legend(("BHE","BLE"),loc="upper left")
    
    fig = plt.figure("Raw data of N-S component",dpi=300)
    axs = fig.subplots()
    axs.set_title("Raw BHN and Raw BLN data",)
    axs.set_ylabel("Counts")
    axs.grid(True)
    axs.plot((BHN.data[-1000:]-np.mean(BHN.data[-1000:])),linewidth=0.8)
    axs.plot(((BLN.data[-1000:]-np.mean(BLN.data[-1000:]))*4),linewidth=0.8)
    axs.legend(("BHN","BLN"),loc="upper left")

    fig = plt.figure("Comparison of corrected BHE data and raw BLE data",dpi=300)
    axs = fig.subplots()
    axs.set_title("Corrected BHE and Raw BLE data",)
    axs.set_ylabel("Counts",)
    axs.grid(True)
    axs.plot((new_bhe.data[-1000:]-np.mean(new_bhe.data[-1000:])),linewidth=0.8)
    axs.plot(((BLE.data[-1000:]-np.mean(BLE.data[-1000:]))*4),linewidth=0.8)
    axs.legend(("NEW_BHE","BLE"),loc="upper left",)

    fig = plt.figure("Comparison of corrected BHN data and raw BLN data",dpi=300)
    axs = fig.subplots()
    axs.grid(True)
    axs.set_title("Corrected BHN and Raw BLN data",)
    axs.set_ylabel("Counts",)
    axs.plot((new_bhn.data[-1000:]-np.mean(new_bhn.data[-1000:])),linewidth=0.8)
    axs.plot(((BLN.data[-1000:]-np.mean(BLN.data[-1000:]))*4),linewidth=0.8)
    axs.legend(("NEW_BHN","BLN"),loc="upper left")
    plt.tight_layout()
    plt.show()

# Furthermore display
def additional_effect(BLE,BLN,new_bhe,new_bhn):
    new_bhe = new_bhe.filter("bandpass",freqmin=0.1,freqmax=0.5)
    new_bhn = new_bhn.filter("bandpass",freqmin=0.1,freqmax=0.5)
    BLE = BLE.filter("bandpass",freqmin=0.1,freqmax=0.5)
    BLN = BLN.filter("bandpass",freqmin=0.1,freqmax=0.5)

    fig= plt.figure("Comparison of corrected BHE data and raw BLE data after filter",dpi=300)
    axs = fig.subplots()
    axs.grid(True)
    axs.set_title("Filtered NEW_BHE and Filtered BLE data ",)
    axs.plot((new_bhe.data[-1000:]-np.mean(new_bhe.data[-1000:])),'x',linewidth=0.8)
    axs.plot(((BLE.data[-1000:]-np.mean(BLE.data[-1000:]))*4),linewidth=0.8)
    axs.legend(("NEW_BHE_FILTERED","BLE_FILTERED"),loc="upper left",)
    
    fig = plt.figure("Comparison of corrected BHN data and raw BLN data after filter",dpi=300)
    axs = fig.subplots()
    axs.grid(True)
    axs.set_title("Filtered NEW_BHN and Filtered BLN data ",)
    axs.plot((new_bhn.data[-1000:]-np.mean(new_bhn.data)),'x',linewidth=0.8)
    axs.plot(((BLN.data[-1000:]-np.mean(BLN.data[-1000:]))*4),linewidth=0.8)
    axs.legend(("NEW_BHN_FILTERED","BLN_FILTERED"),loc="upper left",)
    plt.tight_layout()
    plt.show()

# Calculate the Root Mean Squared Error (RMSE)
def calculate_rmse(BLE,BLN,BHE,BHN):
    rmse_e = np.square(np.sum((BLE.data-np.mean(BLE.data) - (BHE.data-np.mean(BHE))/4)**2)/len(BLE.data))
    rmse_n = np.square(np.sum((BLN.data-np.mean(BLN.data) - (BHN.data-np.mean(BHN))/4)**2)/len(BLN.data))

    rmse_e_new = np.square(np.sum((BLE.data-np.mean(BLE.data) - (new_bhe.data-np.mean(new_bhe))/4)**2)/len(BLE.data))
    rmse_n_new = np.square(np.sum((BLN.data-np.mean(BLN.data) - (new_bhn.data-np.mean(new_bhn))/4)**2)/len(BLN.data))

    df = pd.DataFrame(data = np.corrcoef([BLE.data, BLN.data, BHE.data, BHN.data, new_bhe.data,new_bhn.data]), \
                      columns= ["BLE", "BLN", "BHE", "BHN", "new_BHE","new_BHN"], \
                      index = ["BLE", "BLN", "BHE", "BHN", "new_BHE","new_BHN"] )
    df.to_csv(r"corrcoef.csv")

    print(np.corrcoef([BLE.data, BLN.data, BHE.data, BHN.data, new_bhe.data,new_bhn.data]))
    print(rmse_e, )
    print(rmse_n, )
    print(rmse_e_new, )
    print(rmse_n_new)


# main
if __name__ == '__main__':
    try:
        ST = readfile()  
    except (RuntimeError, TypeError, NameError):
        print("error occured when open data!")

    ST.filter("highpass",freq=0.1)
    BHE,BHN,BLE,BLN = process(ST)
    # BHE.data = BHE.data*(-1)  ## Circuit connection is reversed in lab's device, so the data need to multiply by -1.
    ca,sa = calculate_parameters(BHE,BHN,BLE,BLN)
    rec_ca,rec_sa = r_rebuild(ca,sa)
    angle = getangle(rec_ca,rec_sa)
    try:
        angle_1 = getangle(ca,sa)
    except:
        print("calculation error!")
    else:
        print("未重构angle1:",angle_1*180/np.pi,"重构angle:",angle*180/np.pi)

    new_bhe,new_bhn = correction(BHE, BHN ,angle)
    effect(BHE,BHN,BLE,BLN,new_bhe,new_bhn)
    additional_effect(BLE,BLN,new_bhe,new_bhn)
    plothist(ca,sa)
##    illustration(ca,sa)
    
    s_1,s_2=[],[]
    for n in range(100):
        s_1.append(np.mean(random.sample(list(ca),10)))
        s_2.append(np.mean(random.sample(list(rec_ca),10)))
    plt.figure("100 tests",dpi=300)
    plt.plot(s_1,"x",s_2,".",)
    plt.title("Results of 100 tests",)
    plt.legend(("Average of raw data sample","Average of processed data sample"),)
    plt.tight_layout()
    plt.show()
