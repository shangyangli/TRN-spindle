# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:37:07 2023

@author: Shangyang
"""
import math
import os.path
from pathlib import Path

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils

bp.math.enable_x64(True)
bp.math.set_platform('cpu')
bp.math.set_dt(0.01)



    
if __name__ == '__main__1':    #Load frequency (new way)   
    save_path = './data/Fig6/gaussian_conn2/size10_10/uniform/Vr=-63.9/'
    #load data    
    file_name = os.path.join(save_path, 'freq-gjw=0,Vr=-63.9,gT_sigma = 0.8.npz')
    freq11_variable = np.load(file_name)['freq']
    
    file_name = os.path.join(save_path, 'freq-gjw=0.001,Vr=-63.9,gT_sigma = 0.8.npz')
    freq22_variable = np.load(file_name)['freq']
    
    file_name = os.path.join(save_path, 'freq-gjw=0.003,Vr=-63.9,gT_sigma = 0.8.npz')
    freq33_variable = np.load(file_name)['freq']
    
    # Hist diagram of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=3, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data=freq11_variable, kde=False, ax=ax, stat='count', color="#6195C8")
    plt.ylabel('Count',fontsize=30)
    plt.xlabel('Frequency [Hz]',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  # 设置横坐标刻度标签字号
    plt.yticks(fontsize=20)  # 设置纵坐标刻度标签字号
    plt.savefig("./Fig1020/diff_gjw_B/Fig5-B_freq1_dis.png", transparent=True, dpi=500)    
    plt.show()
    
    # Heatmap of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    freq11_variable1 = freq11_variable.reshape((10,10))
    ax = sns.heatmap(freq11_variable1,cbar=False,cmap='plasma',linewidths=.5,linecolor='b')  
    cb = ax.figure.colorbar(ax.collections[0]) 
    cb.ax.tick_params(labelsize=28)  
    plt.tick_params(labelsize=30)
    ax.set_xlabel('Neuron ID',fontsize=30)
    ax.set_ylabel('Neuron ID',fontsize=30)
    plt.savefig("./Fig1020/diff_gjw_B/Fig5-B_freq1_heatmap.png", transparent=True, dpi=500)
    plt.show()
    
    # Hist diagram of freq22_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=3, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data=freq22_variable, kde=False, ax=ax, stat='count', color="#6195C8")
    plt.ylabel('Count',fontsize=30)
    plt.xlabel('Frequency [Hz]',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2) 
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.savefig("./Fig1020/diff_gjw_B/Fig5-B_freq2_dis.png", transparent=True, dpi=500)    
    plt.show()
    
    # Heatmap of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    freq22_variable1 = freq22_variable.reshape((10,10))
    ax = sns.heatmap(freq22_variable1,cbar=False,cmap='plasma',linewidths=.5,linecolor='b')  #cmap='YlGnBu'
    cb = ax.figure.colorbar(ax.collections[0]) 
    cb.ax.tick_params(labelsize=28)  
    plt.tick_params(labelsize=30)
    ax.set_xlabel('Neuron ID',fontsize=30)
    ax.set_ylabel('Neuron ID',fontsize=30)
    #plt.tight_layout
    #ax.set_zticks([7.0, 7.2, 7.4]) 
    cb.set_ticks([7.0, 7.2, 7.4])
    plt.savefig("./Fig1020/diff_gjw_B/Fig5-B_freq2_heatmap.png", transparent=True, dpi=500)
    plt.show()
    
    # Hist diagram of freq22_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=3, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data=freq33_variable, kde=False, ax=ax, stat='count', color="#6195C8")
    plt.ylabel('Count',fontsize=30)
    plt.xlabel('Frequency [Hz]',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.savefig("./Fig1020/diff_gjw_B/Fig5-B_freq3_dis.png", transparent=True, dpi=500)    
    plt.show()
    
    # Heatmap of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    freq33_variable1 = freq33_variable.reshape((10,10))
    ax = sns.heatmap(freq33_variable1,cbar=False,cmap='plasma',linewidths=.5,linecolor='b')  #cmap='YlGnBu'
    cb = ax.figure.colorbar(ax.collections[0]) 
    cb.ax.tick_params(labelsize=28)  
    plt.tick_params(labelsize=30)
    ax.set_xlabel('Neuron ID',fontsize=30)
    ax.set_ylabel('Neuron ID',fontsize=30)
    plt.savefig("./Fig1020/diff_gjw_B/Fig5-B_freq3_heatmap.png", transparent=True, dpi=500)
    plt.show()
    
# LFP visualiztion   gap junction
if __name__ == '__main__1':
  base_path = './data/Fig6/gaussian_conn2/size10_10/uniform/Vr=-63.9/'
  save_path = './data/Fig6/gaussian_conn2/size10_10/uniform/Vr=-63.9/'
  sns.set_style('white')

  gjws = [0 ,0.001,0.003] 
  Vrs = [-63.9]  
  gT_sigmas = [0.8]  

  for i, gjw in enumerate(gjws):
    file_name = os.path.join(base_path, f'gjw={gjw},Vr=-63.9,gT_sigma = 0.8.npz')
    # print(file_name)
    my_file = Path(file_name)
    if my_file.is_file():
      print(file_name)
      fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
      ax = fig.add_subplot(gs[0, 0])
      lfp = np.load(file_name)['lfp']
      time_range = np.arange(len(lfp)) * bm.get_dt()

      fs = 1e3 / bm.get_dt()
      window_sec = 2 / 0.5
      filtered_lfp = utils.bandpass(lfp, 1, 30, fs, corners=3)
      lfp_freqs, psd = utils.bandpower(filtered_lfp, fs, window_sec)
      idx = np.argmax(psd)
      freq = lfp_freqs[idx]
      power = psd[idx]
      
      if i==2:
          plt.plot(time_range[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))],
               filtered_lfp[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))], 'r', label='Filtered TRN', color="#C82423")
      if i==1:
          plt.plot(time_range[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))],
               filtered_lfp[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))], 'r', label='Filtered TRN', color="#F8AC8C")
      if i==0:
          plt.plot(time_range[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))],
               filtered_lfp[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))], 'r', label='Filtered TRN', color="#9AC9DB")
      #plt.ylabel(f'$GJW$={gjw}', fontsize=15)
      #plt.xlim(time_range[int(1 / 4 * len(lfp))], time_range[int(2 / 4 * len(lfp))])
      plt.ylim(-8,12)
      plt.xlim(18000,30000)
      plt.xlabel('Time [ms]', fontsize=30)
      plt.ylabel('LFP', fontsize=30)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['bottom'].set_linewidth(2)  
      ax.spines['left'].set_linewidth(2)    
      plt.tick_params(axis='both', which='major', width=2)  
      plt.tick_params(axis='both', which='minor', width=2)  
      plt.xticks(fontsize=20)  
      plt.yticks(fontsize=20)  
      ax.set_xticks([18000, 24000, 30000])
      
      plt.savefig(f"./Fig1020/Fig4-4_heterTRNGJ_freq{i}.png", dpi=500, transparent=True)

  plt.show()

    
if __name__ == '__main__1':    # gT variance    
    save_path = './data/Fig6/gaussian_conn/size10_10/uniform/Vr=-63.9/'
    #load data
    
    file_name = os.path.join(save_path, 'freq-gjw=0.001,Vr=-63.9,gT_sigma = 0.npz')
    freq11_variable = np.load(file_name)['freq']
    
    file_name = os.path.join(save_path, 'freq-gjw=0.001,Vr=-63.9,gT_sigma = 0.8.npz')
    freq22_variable = np.load(file_name)['freq']
    
    file_name = os.path.join(save_path, 'freq-gjw=0.001,Vr=-63.9,gT_sigma = 2.4.npz')
    freq33_variable = np.load(file_name)['freq']
    
    # Hist diagram of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=3, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data=freq11_variable, kde=False, ax=ax, stat='count', color="#6195C8")
    plt.ylabel('Count',fontsize=30)
    plt.xlabel('Frequency [Hz]',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)   
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20) 
    plt.savefig("./Fig1020/diff_gT_B/Fig5-B_freq1_dis.png", transparent=True, dpi=500)    
    plt.show()
    
    # Heatmap of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    freq11_variable1 = freq11_variable.reshape((10,10))
    ax = sns.heatmap(freq11_variable1,cbar=False,cmap='plasma',linewidths=.5,linecolor='b')  #cmap='YlGnBu'
    cb = ax.figure.colorbar(ax.collections[0]) 
    cb.ax.tick_params(labelsize=28)  
    plt.tick_params(labelsize=30)
    ax.set_xlabel('Neuron ID',fontsize=30)
    ax.set_ylabel('Neuron ID',fontsize=30)
    plt.savefig("./Fig1020/diff_gT_B/Fig5-B_freq1_heatmap.png", transparent=True, dpi=500)
    plt.show()
    
    # Hist diagram of freq22_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=3, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data=freq22_variable, kde=False, ax=ax, stat='count', color="#6195C8")
    plt.ylabel('Count',fontsize=30)
    plt.xlabel('Frequency [Hz]',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20) 
    plt.savefig("./Fig1020/diff_gT_B/Fig5-B_freq2_dis.png", transparent=True, dpi=500)    
    plt.show()
    
    # Heatmap of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    freq22_variable1 = freq22_variable.reshape((10,10))
    ax = sns.heatmap(freq22_variable1,cbar=False,cmap='plasma',linewidths=.5,linecolor='b')  #cmap='YlGnBu'
    cb = ax.figure.colorbar(ax.collections[0]) 
    cb.ax.tick_params(labelsize=28)  
    plt.tick_params(labelsize=30)
    ax.set_xlabel('Neuron ID',fontsize=30)
    ax.set_ylabel('Neuron ID',fontsize=30)
    plt.savefig("./Fig1020/diff_gT_B/Fig5-B_freq2_heatmap.png", transparent=True, dpi=500)
    plt.show()
    
    # Hist diagram of freq22_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=3, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data=freq33_variable, kde=False, ax=ax, stat='count', color="#6195C8")
    plt.ylabel('Count',fontsize=30)
    plt.xlabel('Frequency [Hz]',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)   
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20) 
    plt.savefig("./Fig1020/diff_gT_B/Fig5-B_freq3_dis.png", transparent=True, dpi=500)    
    plt.show()
    
    # Heatmap of freq11_variable
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    freq33_variable1 = freq33_variable.reshape((10,10))
    ax = sns.heatmap(freq33_variable1,cbar=False,cmap='plasma',linewidths=.5,linecolor='b')  #cmap='YlGnBu'
    cb = ax.figure.colorbar(ax.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=28)  
    plt.tick_params(labelsize=30)
    ax.set_xlabel('Neuron ID',fontsize=30)
    ax.set_ylabel('Neuron ID',fontsize=30)
    plt.savefig("./Fig1020/diff_gT_B/Fig5-B_freq3_heatmap.png", transparent=True, dpi=500)
    plt.show()

# LFP visualiztion
if __name__ == '__main__1':   # gT variance  
  base_path = './data/Fig6/gaussian_conn/size10_10/uniform/Vr=-63.9/'
  save_path = './data/Fig6/gaussian_conn/size10_10/uniform/Vr=-63.9/'
  sns.set_style('white')

  gjws = [0.001] 
  Vrs = [-63.9]  
  gT_sigmas = [0, 0.8, 2.4]  

  for i, gT_sigma in enumerate(gT_sigmas):
    file_name = os.path.join(base_path, f'gjw=0.001,Vr=-63.9,gT_sigma = {gT_sigma}.npz')
    # print(file_name)
    my_file = Path(file_name)
    if my_file.is_file():
      print(file_name)
      fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
      ax = fig.add_subplot(gs[0, 0])
      lfp = np.load(file_name)['lfp']
      time_range = np.arange(len(lfp)) * bm.get_dt()

      fs = 1e3 / bm.get_dt()
      window_sec = 2 / 0.5
      filtered_lfp = utils.bandpass(lfp, 1, 30, fs, corners=3)
      lfp_freqs, psd = utils.bandpower(filtered_lfp, fs, window_sec)
      idx = np.argmax(psd)
      freq = lfp_freqs[idx]
      power = psd[idx]
      
      if i==2:
          plt.plot(time_range[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))],
               filtered_lfp[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))], 'r', label='Filtered TRN', color="#9AC9DB")
      if i==1:
          plt.plot(time_range[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))],
               filtered_lfp[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))], 'r', label='Filtered TRN', color="#F8AC8C")
      if i==0:
          plt.plot(time_range[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))],
               filtered_lfp[int(1 / 4 * len(lfp)):int(2 / 4 * len(lfp))], 'r', label='Filtered TRN', color="#C82423")
      #plt.ylabel(f'$GJW$={gjw}', fontsize=15)
      #plt.xlim(time_range[int(1 / 4 * len(lfp))], time_range[int(2 / 4 * len(lfp))])
      plt.ylim(-8,12)
      plt.xlim(18000,30000)
      plt.xlabel('Time [ms]', fontsize=30)
      plt.ylabel('LFP', fontsize=30)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['bottom'].set_linewidth(2)  # 设置横坐标轴线宽为2
      ax.spines['left'].set_linewidth(2)    # 设置纵坐标轴线宽为2
      plt.tick_params(axis='both', which='major', width=2)  # 设置主要刻度线宽度为4
      plt.tick_params(axis='both', which='minor', width=2)  # 设置次要刻度线宽度为4
      plt.xticks(fontsize=20)  # 设置横坐标刻度标签字号
      plt.yticks(fontsize=20)  # 设置纵坐标刻度标签字号
      ax.set_xticks([18000, 24000, 30000])
      
      plt.savefig(f"./Fig1020/Fig4-4_heterTRNgT_freq{i}.png", dpi=500, transparent=True)

  plt.show()
 
if __name__ == '__main__1':   
    # hundred_sine_waves
    neuron_num = 100
    x = np.arange(100, 250, 0.01)
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5/2, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    mu = 1
    np.random.seed(5)
    fss = mu + 0.2 * np.random.random((neuron_num, 1))  # np.random.normal(mu, sigma, neuron_num)
    var = round(fss.std(), 2)
    np.random.seed(1)
    noise = 0 * np.random.random((neuron_num, 1))
    y1 = np.sin(2 * math.pi * fss * x + noise)
    plt.plot(x, y1.sum(axis=0), color="#6195C8")
    plt.ylim(-22., 22.)
    plt.xlim(100, 150)    
    plt.xlabel('Time [ms]', fontsize=30)
    # plt.ylabel('Amplitude', fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  # 设置横坐标刻度标签字号
    plt.yticks(fontsize=20)  # 设置纵坐标刻度标签字号
    plt.savefig("./Fig1020/Fig4-manySinWave.png", dpi=500, transparent=True)
    plt.show() 
    
    # two_sine_wave
    duration = 5
    f1, f2 = 8.5, 9.
    ts = np.arange(0, duration, 0.001)  # s
    a = np.cos(2 * np.pi * f1 * ts)
    b = np.cos(2 * np.pi * f2 * ts)
    
    # two single wave
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5/2, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(ts, a, label=r'$y_1$')
    plt.plot(ts, b, label=r'$y_2$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend() 
    plt.xlabel('Time [ms]', fontsize=30)
    # plt.ylabel('Amplitude', fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.savefig("./Fig1020/Fig4-twoSinWave1.png", dpi=500, transparent=True)
    plt.show() 
    
    # two single wave sum
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5/2, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(ts, a + b, label=r'$y$', color="#6195C8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Time [ms]', fontsize=30)
    # plt.ylabel('Amplitude', fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.savefig("./Fig1020/Fig4-twoSinWave2.png", dpi=500, transparent=True)
    plt.show() 
    
