# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:58:56 2023

@author: Shangyang
"""

import os.path

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

import utils
import neus_and_syns
import spindle_detection
import multiprocessing
import pandas as pd
import seaborn as sns
from mne.filter import filter_data
from matplotlib.ticker import MaxNLocator
import spindle_detection

#bp.math.enable_x64(True)
bp.math.set_platform('cpu')
bp.math.set_dt(0.01)


plt.rcParams.update({"font.size": 15})

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan']

#Fig2-A
def fig_SR():
    lfo_data_fn = r'./data/Fig3_1/Vr=-73/gjw=0.001/'
    lfo_data = np.load(lfo_data_fn + r'\membrane_potential-gjw=0.001,Vr=-73,gT_sigma = 0.8.npz')['membrane_potential']
    lfo_sps = np.load(lfo_data_fn + r'\spikes-gjw=0.001,Vr=-73,gT_sigma = 0.8.npz')['spikes']

    lfo_times = np.arange(lfo_data.shape[0]) * 1e-5
    lfo_spikes = np.logical_and(lfo_data[:-1] < 0, lfo_data[1:] >= 0)
    
    def visualize_lfo(data, spikes, t_start: float, t_end: float, sf=1e5,
                      all_ids=(0, 19, 48, 78), color="#9AC9DB"):   
        start = int(t_start * 1e5)
        end = int(t_end * 1e5)
      
        times = np.arange(data.shape[0]) / sf
      
        fig, gs = bp.visualize.get_figure(8, 1, 4/8, 6.)
        # Rater Plot
        ax1 = fig.add_subplot(gs[0:8, 0])
        iis, jjs = np.where(spikes)
        plt.plot(times[iis], jjs, 'k.', markersize=2.5, color=color)
        plt.ylabel('Neuron Index', fontsize=30) 
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlim(t_start, t_end)
        ax1.yaxis.set_major_locator(MaxNLocator(1))
        ax1.set_xticks([5, 10, 15])
        # plt.yticks([])
        # plt.xticks([])
        plt.xlabel('Time [s]', fontsize=30) 
        ax1.spines['bottom'].set_linewidth(2)  
        ax1.spines['left'].set_linewidth(2)    
        plt.tick_params(axis='both', which='major', width=2)  
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20) 
        plt.yticks(fontsize=20)  
        
        plt.savefig('./Fig3/lfo-sps-potentials.png', dpi=500, transparent=True)
        
        
        
        fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
        # LFP
        data_filt = filter_data(np.mean(data, axis=1), sf, 2, 7, method='fir', verbose=0)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.plot(times[start: end], data_filt[start: end], label='Filtered LFP', color=color, linewidth=2.5)
        plt.ylabel('LFP', fontsize=30)
        plt.xlabel('Time [s]', fontsize=30)
        ax1.set_xlim(t_start, t_end)
        ax1.spines['top'].set_visible(False)
        ax1.set_xticks([5, 10, 15])
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(2)  
        ax1.spines['left'].set_linewidth(2)    
        plt.tick_params(axis='both', which='major', width=2)  
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        plt.ylim(-11, 11)
        
        plt.savefig('./Fig3/lfo-lfp.png', dpi=500, transparent=True)
        
        fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
        # Power
        ax1 = fig.add_subplot(gs[0, 0])
        window_sec = 2 / 0.5
        lfp_freqs, psd = utils.bandpower(data_filt, sf, window_sec)
        length = int(30 * window_sec)
        plt.plot(lfp_freqs[:length], psd[:length], color=color, linewidth=2.5)
        plt.xlabel('Frequency [Hz]', fontsize=30)
        plt.ylabel('Power', fontsize=30)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlim(0., 15.)
        ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax1.spines['bottom'].set_linewidth(2) 
        ax1.spines['left'].set_linewidth(2)    
        plt.tick_params(axis='both', which='major', width=2)  
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        
        plt.savefig('./Fig3/lfo-power.png', dpi=500, transparent=True)
      
        plt.show()
      
    visualize_lfo(lfo_data, lfo_sps, 5., 15.)


# Fig2-B
def fig_spindle():
    filename = r'./data/Fig3_1/spindle/example1.npz'
    data = np.load(filename)['V']
    times = np.arange(data.shape[0]) * 1e-5
    spikes = np.logical_and(data[:-1] < 0, data[1:] >= 0)

    sps = spindle_detection.spindles_detect(
        np.mean(data, axis=1),
        sf=1e5,
        thresh={'rel_pow': 0.9, 'corr': 0.9, 'rms': 0.05},
        corr_window=0.1,
        corr_step=0.05,
        freq_sp=(6, 10),
        show=True,
    )
    
    def visualize_spindle(sps, t_start: float, t_end: float):
        start = int(t_start * 1e5)
        end = int(t_end * 1e5)
        sf = 1e5
    
        sps_summary = sps.summary()
    
        times = np.arange(data.shape[0]) / sf
        spikes = np.logical_and(data[:-1] < 0, data[1:] >= 0)
    
        fig, gs = bp.visualize.get_figure(8, 1, 4/8, 6.)
        # Rater Plot
        ax1 = fig.add_subplot(gs[0:8, 0])
        iis, jjs = np.where(spikes)
        plt.plot(times[iis], jjs, 'k.', markersize=2.5, color="#F8AC8C")
        plt.ylabel('Neuron Index', fontsize=30)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlim(t_start, t_end)
        ax1.set_xticks([5, 10, 15])
        ax1.yaxis.set_major_locator(MaxNLocator(1))
        plt.xlabel('Time [s]', fontsize=30)
        ax1.spines['bottom'].set_linewidth(2)  
        ax1.spines['left'].set_linewidth(2)   
        plt.tick_params(axis='both', which='major', width=2)  
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        plt.savefig('./Fig3/spindle-sps-potentials.png', dpi=500, transparent=True)
    
        fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
        # LFP
        ax1 = fig.add_subplot(gs[0, 0])
        plt.plot(times[start: end], sps._data_filt[0, start: end], 'k', label='Filtered LFP', color="#F8AC8C", linewidth=2.5)
        starts = np.asarray(sps_summary.Start.to_numpy() * 1e5, dtype=int)
        ends = np.asarray(sps_summary.End.to_numpy() * 1e5, dtype=int)
        for i, (s, e) in enumerate(zip(starts, ends)):
          if i == 0:
            plt.plot(times[s: e], sps._data_filt[0, s: e], 'g', label='Detected spindle', color= "#008F7A", linewidth=2.5)
          else:
            plt.plot(times[s: e], sps._data_filt[0, s: e], 'g')
        plt.ylabel('LFP', fontsize=30)
        plt.xlabel('Time [s]', fontsize=30)
        plt.legend(fontsize=10)
        ax1.set_yticks([-5, 0, 5])
        ax1.set_xticks([5, 10, 15])
        ax1.set_xlim(t_start, t_end)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.ylim(-6, 6)
        ax1.spines['bottom'].set_linewidth(2)  
        ax1.spines['left'].set_linewidth(2)    
        plt.tick_params(axis='both', which='major', width=2)  
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        plt.savefig('./Fig3/spindle-lfp.png', dpi=500, transparent=True)
        
        
        # Power
        fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
        ax2 = fig.add_subplot(gs[0, 0])
        window_sec = 2 / 0.5
        lfp_freqs, psd = utils.bandpower(sps._data_filt[0], sf, window_sec)
        length = int(30 * window_sec)
        plt.plot(lfp_freqs[:length], psd[:length], color="#F8AC8C", linewidth=2.5)
        plt.xlabel('Frequency [Hz]', fontsize=30)
        plt.ylabel('Power', fontsize=30)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(0., 15.)
        ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax1.set_yticks([0, 5])
        ax2.spines['bottom'].set_linewidth(2)  
        ax2.spines['left'].set_linewidth(2)    
        plt.tick_params(axis='both', which='major', width=2)  
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
    
        plt.savefig('./Fig3/spindle-power.png', dpi=500, transparent=True)
    
        plt.show()

    visualize_spindle(sps, 5, 15)
 
    
 
# Fig2_C    
def fig_IR():
    ir_sps_data_fn = r'./data/Fig3_1/irregular_spiking/gT_sigma=0.8,Vr=-60.0,gjw=0.001,b=0.5,rho_p=0.01,E_KL=-100.0,g_L=0.05,E_L=-60.0,gaba_g_max=0.01,amap_g_max=0.01.npz'
    ir_sps_data = np.load(ir_sps_data_fn)
    ir_sps_V = ir_sps_data['potentials']
    ir_sps_S = ir_sps_data['spikes']

    def visualize_ir(data, spikes, t_start: float, t_end: float, sf=1e5,
                     all_ids=(0, 19, 48, 78)):
        start = int(t_start * 1e5)
        end = int(t_end * 1e5)
      
        times = np.arange(data.shape[0]) / sf
      
        fig, gs = bp.visualize.get_figure(2, 1, 4/2, 6.)
        # Rater Plot
        ax1 = fig.add_subplot(gs[0:2, 0])
        iis, jjs = np.where(spikes)
        plt.plot(times[iis], jjs, 'k.', markersize=2.5, color="#C82423")
        plt.ylabel('Neuron Index', fontsize=30)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlim(t_start, t_end)
        ax1.yaxis.set_major_locator(MaxNLocator(1))
        plt.xlabel('Time [s]', fontsize=30)
        ax1.set_xticks([5, 10, 15])
        ax1.spines['bottom'].set_linewidth(2)  
        ax1.spines['left'].set_linewidth(2)    
        plt.tick_params(axis='both', which='major', width=2)  
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        plt.savefig('./Fig3/ir-sps-potentials.png', dpi=500, transparent=True)
      
        fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
        # LFP
        data_filt = filter_data(np.mean(data, axis=1), sf, 2, 40, method='fir', verbose=0)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.plot(times[start: end], data_filt[start: end], 'k', label='Filtered LFP', color="#C82423", linewidth=2.5)
        plt.ylabel('LFP', fontsize=30)
        plt.xlabel('Time [s]', fontsize=30)
        # plt.legend(fontsize=10)
        ax1.set_xlim(t_start, t_end)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(2)  
        ax1.spines['left'].set_linewidth(2)    
        plt.tick_params(axis='both', which='major', width=2) 
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        plt.ylim(-3, 3)
        ax1.set_xticks([5, 10, 15])
        plt.savefig('./Fig3/ir-lfp.png', dpi=500, transparent=True)
        
        
        # Power
        fig, gs = bp.visualize.get_figure(1, 1, 2, 6.)
        ax2 = fig.add_subplot(gs[0, 0])
        window_sec = 2 / 0.5
        lfp_freqs, psd = utils.bandpower(data_filt, sf, window_sec)
        length = int(30 * window_sec)
        plt.plot(lfp_freqs[:length], psd[:length], color="#C82423", linewidth=2.5)
        plt.xlabel('Frequency [Hz]', fontsize=30)
        plt.ylabel('Power', fontsize=30)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_linewidth(2)  
        ax2.spines['left'].set_linewidth(2)   
        plt.tick_params(axis='both', which='major', width=2) 
        plt.tick_params(axis='both', which='minor', width=2)  
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        ax2.set_xlim(0., 15.)
        ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        plt.ylim(0, 0.1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
      
        plt.savefig('./Fig3/ir-power.png', dpi=500, transparent=True)
      
        plt.show()

    visualize_ir(ir_sps_V, ir_sps_S, 5., 15., sf=1e5, all_ids=(48, 55, 73))





## Fig2-J
# Oscillation power region
def phase_diagram():
  # visualization
  fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=1.2, col_len=6)
  ax = fig.add_subplot(gs[0, 0])
  plt.subplots_adjust(bottom=0.15, top=0.85)  
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.gca().spines['left'].set_visible(False)
  plt.gca().spines['bottom'].set_visible(False)
  # 明确指定横坐标轴的刻度值
  plt.ylim(3, 8)
  plt.xlim(0, 0.02)
  plt.xticks([0.0, 0.004, 0.008, 0.012, 0.016, 0.02])

  plt.xticks(fontsize=20)  
  plt.yticks([])  # 移除纵坐标刻度标签
  plt.tick_params(axis='x', which='both', bottom=False, top=False)
  plt.gca().spines['left'].set_visible(False)
  plt.gca().spines['bottom'].set_visible(False)
  plt.tick_params(axis='both', which='major', width=0)  
  plt.xticks(fontsize=20)  

  x1 = np.linspace(0, 0.004, 20)
  y1 = np.ones(20) * 3
  y2 = np.ones(20) * 8
  plt.fill_between(x1, y1, y2, alpha=.5, linewidth=0, color='#C82423')

  x2 = np.linspace(0.004, 0.007, 20)
  y3 = np.ones(20) * 3
  y4 = np.ones(20) * 8
  plt.fill_between(x2, y3, y4, alpha=.5, linewidth=0, color='#F8AC8C')

  x3 = np.linspace(0.007, 0.02, 20)
  y5 = np.ones(20) * 3
  y6 = np.ones(20) * 8
  plt.fill_between(x3, y5, y6, alpha=.5, linewidth=0, color='#9AC9DB') 

  plt.xlabel('$g_{KL}$', fontsize=30)
  plt.savefig("./Fig3/Fig3-4_state_diagram1.png", dpi=500, transparent=True)
  plt.show()
  
def lfpPatternOscillotion(t_start: float, t_end: float, sf=1e5,
                  all_ids=(0, 19, 48, 78), color="#9AC9DB"):
    lfo_data_fn = r'./data/Fig3_1/Vr=-73/gjw=0.001/'
    lfo_data = np.load(lfo_data_fn + r'\membrane_potential-gjw=0.001,Vr=-73,gT_sigma = 0.8.npz')['membrane_potential']
    data = lfo_data
    times = np.arange(data.shape[0]) / sf
    start = int(t_start * 1e5)
    end = int(t_end * 1e5)
    
    fig, gs = bp.visualize.get_figure(1, 1, 4, 4)
    # LFP
    data_filt = filter_data(np.mean(data, axis=1), sf, 2, 7, method='fir', verbose=0)
    ax1 = fig.add_subplot(gs[0, 0])
    plt.plot(times[start: end], data_filt[start: end], label='Filtered LFP', color=color, linewidth=3)
    #plt.ylabel('LFP [mV]', fontsize=20)
    plt.xlabel('Time [s]', fontsize=20)
    ax1.set_xlim(t_start, t_end)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(2)  
    ax1.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.ylim(-11, 11)
    
    plt.savefig('./Fig3/Fig2-lfo-lfp.png', dpi=500, transparent=True)
  
def lfpPatternSpindle(t_start: float, t_end: float):
    filename = r'./data/Fig3_1/spindle/example1.npz'
    data = np.load(filename)['V']
    
    sps = spindle_detection.spindles_detect(
        np.mean(data, axis=1),
        sf=1e5,
        thresh={'rel_pow': 0.9, 'corr': 0.9, 'rms': 0.05},
        corr_window=0.1,
        corr_step=0.05,
        freq_sp=(6, 10),
        show=True,
    )
    
    start = int(t_start * 1e5)
    end = int(t_end * 1e5)
    sf = 1e5
    times = np.arange(data.shape[0]) / sf
    
    fig, gs = bp.visualize.get_figure(1, 1, 4, 4.)
    # LFP
    ax1 = fig.add_subplot(gs[0, 0])
    plt.plot(times[start: end], sps._data_filt[0, start: end], 'k', color="#F8AC8C", linewidth=3)
    #plt.ylabel('LFP [mV]', fontsize=20)
    plt.xlabel('Time [s]', fontsize=20)
    plt.legend(fontsize=10)
    ax1.set_yticks([-5, 0, 5])
    ax1.set_xlim(t_start, t_end)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.ylim(-5, 5)
    ax1.spines['bottom'].set_linewidth(2)  
    ax1.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.savefig('./Fig3/fig2-spindle-lfp.png', dpi=500, transparent=True)
    
def lfpPatternIR(t_start: float, t_end: float, sf=1e5):
    ir_sps_data_fn = r'./data/Fig3_1/irregular_spiking/gT_sigma=0.8,Vr=-60.0,gjw=0.001,b=0.5,rho_p=0.01,E_KL=-100.0,g_L=0.05,E_L=-60.0,gaba_g_max=0.01,amap_g_max=0.01.npz'
    ir_sps_data = np.load(ir_sps_data_fn)
    ir_sps_V = ir_sps_data['potentials']
    data = ir_sps_V
    
    start = int(t_start * 1e5)
    end = int(t_end * 1e5)
    times = np.arange(data.shape[0]) / sf
    
    fig, gs = bp.visualize.get_figure(1, 1, 4, 4.)
    
    data_filt = filter_data(np.mean(data, axis=1), sf, 2, 40, method='fir', verbose=0)
    ax1 = fig.add_subplot(gs[0, 0])
    plt.plot(times[start: end], data_filt[start: end], 'k', label='Filtered LFP', color="#C82423", linewidth=3)
    #plt.ylabel('LFP [mV]', fontsize=20)
    plt.xlabel('Time [s]', fontsize=20)
    # plt.legend(fontsize=10)
    ax1.set_xlim(t_start, t_end)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(2)  
    ax1.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2) 
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.ylim(-3.5, 3.5)
    plt.savefig('./Fig3/fig2-ir-lfp.png', dpi=500, transparent=True)
    

## Fig2-K L
# show data
def model_exp_results(filename=None):
  def filter(a, remove_list=('0.6', '0.8',)):
    for r in remove_list:
      a = a.groupby(a.sigma == r).get_group(False)
    return a

  base_path = './data/Fig4/'  
  intervals = pd.read_pickle(os.path.join(base_path, 'intervals.pkl'))
  spindles = pd.read_pickle(os.path.join(base_path, 'spindles.pkl'))
  intervals = filter(intervals, ('0.6', '0.8'))
  spindles = filter(spindles, ('0.6', '0.8'))
  print(intervals)
  print(spindles)

  # cycles #
  # ------ #
  fig, gs = bp.visualize.get_figure(2, 1, 4.5/2, 6.)
  ax = fig.add_subplot(gs[0, 0])
  with open(base_path + 'Fig4_1.csv', 'r') as f:
    lines = f.read().strip().split('\n')
    lines = [line.split(',') for line in lines]
    cycles = [int(line[0]) for line in lines]
    numbers = np.asarray([float(line[1]) for line in lines])
  plt.bar(cycles, numbers / numbers.sum(), color="#FE817D")
  #plt.ylabel("Probability", fontsize=20)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_linewidth(2)  
  ax.spines['left'].set_linewidth(2)    
  ax.set_yticks([0, 0.1])
  plt.tick_params(axis='both', which='major', width=2)  
  plt.tick_params(axis='both', which='minor', width=2)  
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  
  plt.xticks([5, 10, 15, 20, 25])


  ax = fig.add_subplot(gs[1, 0])
  cycles = spindles.Oscillations.to_numpy()
  counts, bins = np.histogram(cycles, bins=21)
  print(bins)
  plt.bar(bins[1:], counts / counts.sum(), color="#81B8DF")
  plt.yticks([0., 0.05, 0.10])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  #plt.ylabel("Probability", fontsize=20)
  plt.xlabel('Cycles', fontsize=30)
  ax.spines['bottom'].set_linewidth(2)  
  ax.spines['left'].set_linewidth(2)    
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  
  ax.set_yticks([0, 0.1])
  plt.tick_params(axis='both', which='major', width=2)  
  plt.tick_params(axis='both', which='minor', width=2)  
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  

  if filename:
    plt.savefig(f'./Fig3/{filename}-cycle.png', dpi=500, transparent=True)

  # Duration #
  # -------- #
  fig, gs = bp.visualize.get_figure(2, 1, 4.5/2, 6.)
  with open(base_path + 'Fig4_2.csv', 'r') as f:
    lines = f.read().strip().split('\n')
    lines = [line.split(',') for line in lines]
    lengths = np.asarray([int(line[0]) for line in lines])
    ratios = np.asarray([float(line[1]) for line in lines])
  ax = fig.add_subplot(gs[0, 0])
  print(len(ratios))
  plt.bar(lengths / 1e3, ratios / ratios.sum(), width=0.08, color="#FE817D")
  #plt.ylabel("Probability", fontsize=20)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_linewidth(2)  
  ax.spines['left'].set_linewidth(2)    
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  
  ax.set_yticks([0, 0.1])

  ax = fig.add_subplot(gs[1, 0])
  durations = spindles.Duration.to_numpy()
  counts, bins = np.histogram(durations, bins=21)
  print(bins)
  plt.bar(bins[1:], counts / counts.sum(), width=0.1, color="#81B8DF")
  plt.xlabel("Duration [s]", fontsize=30)
  #plt.ylabel("Probability", fontsize=20)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_linewidth(2)  
  ax.spines['left'].set_linewidth(2)   
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  
  ax.set_yticks([0, 0.1])

  if filename:
    plt.savefig(f'./Fig3/{filename}-duration.png', dpi=500, transparent=True)
  plt.show()





if __name__ == '__main__':
  
  lfpPatternOscillotion(7, 11, sf=1e5, all_ids=(0, 19, 48, 78), color="#9AC9DB")
  lfpPatternSpindle(7, 11)
  lfpPatternIR(7, 11)

