# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:01:06 2023

@author: Shangyang
"""

import os.path

import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import seaborn as sns

bp.math.enable_x64(True)
bp.math.set_platform('cpu')
bp.math.set_dt(0.01)

def spikeDataTransfrom(base_path, file_name):
    "Get new spike data according to the average ISI."
    file_name = os.path.join(base_path, file_name)
    spike_datas = np.load(file_name)['spike_datas']
    
    # Get average ISI 
    spike_sum = np.sum(spike_datas, axis=1)
    spike_sum = np.int64(spike_sum>0)
    average_ISI = int(spike_datas.shape[0]/sum(spike_sum))
    print(f'Average ISI is {average_ISI/100} ms')
    
    # Division of data based on average_ISI  
    aaa = np.linspace(0,int(sum(spike_sum)),int(sum(spike_sum)
                                                  )).astype(int).repeat(average_ISI+1)
    spike_class = aaa[:-(len(aaa)-len(spike_sum))]
    spike_synthesis = np.bincount(spike_class, spike_sum)
    spike_synthesis[0] = 0
    spike_synthesis[-1] = 0
    return average_ISI, spike_synthesis 

def getSizeLifetime(spike_synthesis):
    size = 0
    lifetime = 0
    sizes = []
    lifetimes = []
    for i in range(len(spike_synthesis)-1):
        if spike_synthesis[i]==0 and spike_synthesis[i+1]!=0:
            size = size + int(spike_synthesis[i+1])
            lifetime = lifetime + 1
        elif spike_synthesis[i]!=0 and spike_synthesis[i+1]!=0:
            size = size + int(spike_synthesis[i+1])
            lifetime = lifetime + 1
        elif spike_synthesis[i]!=0 and spike_synthesis[i+1]==0:
            sizes.append(size)
            lifetimes.append(lifetime)
            size = 0
            lifetime = 0
    return sizes,lifetimes

def hisPlot(data, xlabel, file_name):
    # Frequency distribution histogram for sizes
    sns.set()
    plt.figure(figsize=(6.5,4.8))
    #sns.set_palette("hls")   
    sns.set_style('white')
    sns.distplot(data, color="steelblue",kde=False,bins=10,hist_kws=dict(edgecolor="k", linewidth=2))    
    plt.ylabel(f'P({xlabel})',fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.tick_params(labelsize=15)
    #plt.savefig("./Size_dis.eps")
    plt.savefig(f"./Fig1020/Spike_data2/FreqDis_{xlabel}_{file_name}.png", transparent=True, dpi=500, bbox_inches='tight')
    #plt.title("FreqDisPlot")
    plt.show()
    
def powerPlot(data, xlabel, file_name, fitline=False,xmin=1, xmax=15):
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    plt.figure(figsize=(10,10))
    #fig2 = powerlaw.plot_pdf(data, color = 'b', linestyle = 'solid', linewidth = 5)
    fig2 = powerlaw.plot_pdf(data, color = 'b', linewidth = 5, linestyle = 'dotted')
    if fitline == True:
        fit.power_law.plot_pdf(color = 'b', linestyle = '--', linewidth = 3, ax = fig2)
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(f'P({xlabel})',fontsize=30)
    plt.tick_params(labelsize=20)
    plt.savefig(f"./Fig1020/Spike_data2/PowerLaw_{xlabel}_{file_name}.png")
    plt.title("PowerLawPlot")
    plt.show()  
    
if __name__ == '__main__1':
    base_path = './data/fig8/Spike_data/gjw=0.001/'
    file_name = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    hisPlot(data=sizes, xlabel='Size',file_name = file_name)
    hisPlot(data=lifetimes, xlabel='Lifetime',file_name = file_name)
    
    base_path = './data/fig8/Spike_data/gjw=0/'
    file_name = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    hisPlot(data=sizes, xlabel='Size',file_name = file_name)
    hisPlot(data=lifetimes, xlabel='Lifetime',file_name = file_name)
    
    base_path = './data/fig8/Spike_data/gjw=0.003/'
    file_name = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    hisPlot(data=sizes, xlabel='Size',file_name = file_name)
    hisPlot(data=lifetimes, xlabel='Lifetime',file_name = file_name)
    
if __name__ == '__main__1':
    base_path = './data/fig8/Spike_data/gjw=0.001/'
    file_name = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    data=sizes 
    data1=sizes
    xlabel='Size'
    file_name = file_name 
    fitline=False
    xmin=2
    xmax=30
    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    
    # size analysis 1
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    powerlaw.plot_pdf(data, color = '#F8AC8C', linewidth = 3, marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    if fitline == True:
        fit.power_law.plot_pdf(color = 'k', linestyle = '-', linewidth = 1, ax = ax)
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(f'P({xlabel})',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  # 设置横坐标轴线宽为2
    ax.spines['left'].set_linewidth(2)    # 设置纵坐标轴线宽为2
    plt.tick_params(axis='both', which='major', width=2)  # 设置主要刻度线宽度为4
    plt.tick_params(axis='both', which='minor', width=2)  # 设置次要刻度线宽度为4
    plt.xticks(fontsize=20)  # 设置横坐标刻度标签字号
    plt.yticks(fontsize=20)  # 设置纵坐标刻度标签字号
    # 设置 x 轴和 y 轴的范围
    plt.xlim(0, 100)
    plt.ylim(0.0005, 0)
    plt.savefig("./Fig1020/Spike_data2/Fig5_size1-1.png", transparent=True, dpi=500)
    plt.show()
    
   
    # size analysis 2
    base_path = './data/fig8/Spike_data/gjw=0/'
    file_name = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    data=sizes
    # size analysis 2  figure
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    powerlaw.plot_pdf(data, color = '#9AC9DB', linewidth = 3, marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    powerlaw.plot_pdf(data1, color = '#F8AC8C', linewidth = 0, #marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    if fitline == True:
        fit.power_law.plot_pdf(color = 'k', linestyle = '-', linewidth = 1, ax = ax)

    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(f'P({xlabel})',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.xlim(0, 100)
    plt.ylim(0.0005, 0)
    plt.savefig("./Fig1020/Spike_data2/Fig5_size2-1.png", transparent=True, dpi=500)
    plt.show()
    
    
    # size analysis 3
    base_path = './data/fig8/Spike_data/gjw=0.003/'
    file_name = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    data=sizes 
    xlabel='Size'
    file_name = file_name    
    # size analysis 3  figure
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    powerlaw.plot_pdf(data, color = '#C82423', linewidth = 3, marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    powerlaw.plot_pdf(data1, color = '#F8AC8C', linewidth = 0, #marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    if fitline == True:
        fit.power_law.plot_pdf(color = 'k', linestyle = '-', linewidth = 1, ax = ax)

    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(f'P({xlabel})',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)   
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.xlim(0, 100)
    plt.ylim(0.0005, 0)
    plt.savefig("./Fig1020/Spike_data2/Fig5_size3-1.png", transparent=True, dpi=500)
    plt.show()
    
if __name__ == '__main__1':
    base_path = './data/fig8/Spike_data/gjw=0.001/'
    file_name = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    data= lifetimes
    xlabel='Lifetime'
    file_name = file_name 
    fitline=True
    xmin=2
    xmax=30
    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    fig,ax = plt.subplots(figsize=(6,4.5))
    #fig2 = powerlaw.plot_pdf(data, color = 'b', linestyle = 'solid', linewidth = 5)
    powerlaw.plot_pdf(data, color = '#B7BF99', linewidth = 5, linestyle = ':', label='Spindle Oscillation', ax=ax)
    if fitline == True:
        fit.power_law.plot_pdf(color = 'k', linestyle = '-', linewidth = 1, ax = ax)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(f'P({xlabel})',fontsize=20)
    plt.tick_params(labelsize=15)
    
    base_path = './data/fig8/Spike_data/gjw=0/'
    file_name = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    plt.figure(figsize=(6,4.5))
    #fig2 = powerlaw.plot_pdf(data, color = 'b', linestyle = 'solid', linewidth = 5)
    data= lifetimes
    xlabel='Lifetime'
    file_name = file_name 
    powerlaw.plot_pdf(data, color = '#0A7373', linewidth = 4, linestyle = ':', ax = ax, label='Noise')
    
    base_path = './data/fig8/Spike_data/gjw=0.003/'
    file_name = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    plt.figure(figsize=(6,4.5))
    #fig2 = powerlaw.plot_pdf(data, color = 'b', linestyle = 'solid', linewidth = 5)
    data= lifetimes
    xlabel='Lifetime'
    file_name = file_name 
    powerlaw.plot_pdf(data, color = '#EDAA25', linewidth = 4, linestyle = ':', ax = ax, label='Synchronous oscillation')
        
    ax.legend(fontsize=12)  # 添加图例
    fig.savefig("./Fig1020/Spike_data2/PowerLaw_dis_lifetime.png", transparent=True, dpi=500, bbox_inches='tight')    
        
    
if __name__ == '__main__1':
    base_path = './data/fig8/Spike_data/gjw=0.001/'
    file_name = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.001,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    data=lifetimes
    data1=lifetimes
    xlabel='Lifetime'
    file_name = file_name 
    fitline=False
    xmin=2
    xmax=30
    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    
    # size analysis 1
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    powerlaw.plot_pdf(data, color = '#F8AC8C', linewidth = 3, marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    if fitline == True:
        fit.power_law.plot_pdf(color = 'k', linestyle = '-', linewidth = 1, ax = ax)
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(f'P({xlabel})',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)   
    plt.tick_params(axis='both', which='major', width=2) 
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.xlim(0, 30)
    plt.ylim(0.0005, 0)
    plt.savefig("./Fig1020/Spike_data2/Fig5_lifetime1-1.png", transparent=True, dpi=500)
    plt.show()
    
   
    # size analysis 2
    base_path = './data/fig8/Spike_data/gjw=0/'
    file_name = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)
    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    print(fit.power_law.alpha)
    alpha, p_value = fit.distribution_compare('power_law', 'exponential') 
    print(f'likelihood = {alpha}, p_value = {p_value}')
    data=lifetimes
    # size analysis 2  figure
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    powerlaw.plot_pdf(data, color = '#9AC9DB', linewidth = 3, marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    powerlaw.plot_pdf(data1, color = '#F8AC8C', linewidth = 0, #marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    if fitline == True:
        fit.power_law.plot_pdf(color = 'k', linestyle = '-', linewidth = 1, ax = ax)

    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(f'P({xlabel})',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2) 
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.xlim(0, 30)
    plt.ylim(0.0005, 0)
    plt.savefig("./Fig1020/Spike_data2/Fig5_lifetime2-1.png", transparent=True, dpi=500)
    plt.show()
    
    
    # size analysis 3
    base_path = './data/fig8/Spike_data/gjw=0.003/'
    file_name = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8'
    file_name1 = 'membrane_potential-gjw=0.003,Vr=-63.9,gT_sigma = 0.8.npz'
    average_ISI, spike_synthesis = spikeDataTransfrom(base_path = base_path, 
                                                      file_name = file_name1)
    sizes,lifetimes =  getSizeLifetime(spike_synthesis=spike_synthesis)    
    fit = powerlaw.Fit(data, discrete=True, xmin=xmin,xmax=xmax) 
    data=lifetimes 
    xlabel='Lifetime'
    file_name = file_name    
    # size analysis 3  figure
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=4.5, col_len=6)
    ax = fig.add_subplot(gs[0, 0])
    powerlaw.plot_pdf(data, color = '#C82423', linewidth = 3, marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    powerlaw.plot_pdf(data1, color = '#F8AC8C', linewidth = 0, #marker='D', markersize=8,
                      linestyle = 'solid', label='Spindle Oscillation', ax=ax)
    if fitline == True:
        fit.power_law.plot_pdf(color = 'k', linestyle = '-', linewidth = 1, ax = ax)

    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(f'P({xlabel})',fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['left'].set_linewidth(2)    
    plt.tick_params(axis='both', which='major', width=2)  
    plt.tick_params(axis='both', which='minor', width=2)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.xlim(0, 30)
    plt.ylim(0.0005, 0)
    plt.savefig("./Fig1020/Spike_data2/Fig5_lifetime3-1.png", transparent=True, dpi=500)
    plt.show()    
    

    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    