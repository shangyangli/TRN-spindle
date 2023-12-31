# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:37:51 2023

@author: Shangyang

Fig1 all figure
"""
import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import neus_and_syns
from neus_and_syns import ReducedTRNv1
plt.rcParams.update({"font.size": 15})
import seaborn as sns
import utils
import os.path
import pandas as pd

bp.math.enable_x64()
bp.math.set_platform('cpu')

# Fig1-B
def fig1B():
    size_ = (10, 10)

    gT_sigma, gT = 0.8, 2.5
    all_gT = utils.uniform1(size=bp.tools.size2num(size_), ranges=(gT - gT_sigma, gT + gT_sigma), seed=57362).value

    gT, gT_sigma = 2.5, 1.
    all_gT = utils.gaussian2(gT, gT_sigma, bp.tools.size2num(size_), ranges=(0, 10.), seed=57362)

    all_gT1 = np.array(all_gT)

    # Hist diagram of gT
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
    ax = fig.add_subplot(gs[0, 0])
    #sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
    sns.histplot(all_gT1, kde=True, ax=ax, stat="count", color="#6195C8")  # 设置stat="density"以显示概率密度
    #sns.histplot(all_gT1, kde=True, ax=ax)   #, color="steelblue"
    plt.xlabel('$g_T$', fontsize=30)
    plt.ylabel('Count', fontsize=30)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)  # 设置横坐标轴线宽为2
    ax.spines['left'].set_linewidth(2)    # 设置纵坐标轴线宽为2
    plt.tick_params(axis='both', which='major', width=2)  # 设置主要刻度线宽度为4
    plt.tick_params(axis='both', which='minor', width=2)  # 设置次要刻度线宽度为4
    plt.xticks(fontsize=20)  # 设置横坐标刻度标签字号
    plt.yticks(fontsize=20)  # 设置纵坐标刻度标签字号
    plt.xlim(0, 6)

    plt.savefig("./Fig3/Figure2-_gT_hist_distribution1.png", transparent=True, dpi=500)



# Fig1-C
def gT_background():
    fig, gs = bp.visualize.get_figure(1, 1, 1.2, 6.)
    ax = fig.add_subplot(gs[0, 0])
    plt.subplots_adjust(bottom=0.15, top=0.85)  # 调整底部和顶部边距
    #sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
    #sns.histplot(all_gT1, color="steelblue", kde=True, ax=ax)
    plt.xlabel('$g_{T}$', fontsize=30)
    plt.ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 6)
    plt.ylim(3, 8)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # 明确指定横坐标轴的刻度值
    plt.yticks([])  # 移除纵坐标刻度标签
    plt.tick_params(axis='both', which='major', width=0)  
    plt.xticks(fontsize=20)  # 设置横坐标刻度标签字号
    
    x1 = np.linspace(0, 2.3, 20)
    y1 = np.ones(20) * 3
    y2 = np.ones(20) * 8
    plt.fill_between(x1, y1, y2, alpha=.5, linewidth=0, color='#9AC9DB')
    
    x2 = np.linspace(2.3, 3.2, 20)
    y3 = np.ones(20) * 3
    y4 = np.ones(20) * 8
    plt.fill_between(x2, y3, y4, alpha=.5, linewidth=0, color="#F8AC8C")
    
    x3 = np.linspace(3.2, 6, 20)
    y5 = np.ones(20) * 3
    y6 = np.ones(20) * 8
    plt.fill_between(x3, y5, y6, alpha=.5, linewidth=0, color='#C82423')

    #plt.tight_layout()
    plt.savefig("./Fig3/Figure2-SI_gT_background.png", transparent=True, dpi=500)

def gT_firingPattern(g_Na=100., g_K=10., b=0.5, rho_p=0., IT_th=-3.,
                     NaK_th=-55., E_KL=-100, g_L=0.06, E_L=-70,
                              Vr= -65., g_T=1, Iexts=0):    #(-80,-70) (-72,-50)  2.33 4
    reduced = neus_and_syns.ReducedTRNv1(size=1, method='exp_auto')

    sns.set(font_scale=1.5)
    sns.set_style("white")

    reduced.NaK_th  = NaK_th
    reduced.g_T  = g_T
    reduced.g_L  = g_L
    reduced.E_L  = E_L
    reduced.E_KL = E_KL
    reduced.b = b
    reduced.rho_p = rho_p
    reduced.g_Na = g_Na
    reduced.g_K = g_K
    reduced.IT_th  = IT_th
    
    
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 4.)
    #ax1_ylim = (-85, 50)

    inputs, duration = bp.inputs.section_input(values=[0., Iexts, 0.], durations=[50., 100., 350.], return_length=True)

    reduced.reset(Vr=-64)
    reduced.g_KL = reduced.suggest_gKL(Vr=Vr, Iext=0., g_T=g_T)
    runnerReduced = bp.DSRunner(reduced, inputs=['input', inputs, 'iter'], monitors=['V', 'spike'])
    runnerReduced(duration=duration)

    start, end = 0, duration
    ax = fig.add_subplot(gs[0, 0])
    sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
    plt.plot(runnerReduced.mon.ts, runnerReduced.mon.V[:, 0], label='reduced', color="#6195C8", linewidth=4)
    plt.xlim(start, end)
    #plt.ylim(ax1_ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-80, -40)
    plt.tick_params(axis='both', which='major', width=3)  # 设置刻度线宽为2
    plt.xticks(fontsize=15)  # 设置横坐标刻度标签字号
    plt.yticks(fontsize=15)  # 设置纵坐标刻度标签字号
    plt.xlabel('Time(ms)')  
    
    plt.savefig("./Fig1/Figure1_gT_wxample2.png", transparent=True, dpi=500)    

# Fig1-D
# -- spike number with different gT -- #
def gT_spikeNumber(init_Vr=-75.,
                   rest_Vr=-65.,
                   duration=4e2,
                   gT=2.5,
                   gT_sigma=1.2,
                   size=(10, 10),
                   method='gaussian',
                   fn=None):
  if method == 'uniform':
    all_gT = utils.uniform1(size=bp.tools.size2num(size), ranges=(gT - gT_sigma, gT + gT_sigma), seed=57362)
  elif method == 'gaussian':
    all_gT = utils.gaussian2(gT, gT_sigma, bp.tools.size2num(size), ranges=(0, 10.), seed=57362)
  else:
    raise ValueError

  trn_pars = dict(b=0.5,
                  rho_p=0.01,
                  IT_th=-3.,
                  NaK_th=-50.,
                  E_KL=-100.,
                  g_L=0.05,
                  E_L=-77.)

  # set parameters of reduced TRN model
  group = neus_and_syns.ReducedTRNv1(size=all_gT.size, method='exp_auto')
  group.rho_p = trn_pars.get('rho_p')
  group.b = trn_pars.get('b')
  group.g_L = trn_pars.get('g_L')
  group.E_L = trn_pars.get('E_L')
  group.E_KL = trn_pars.get('E_KL')
  group.IT_th = trn_pars.get('IT_th')
  group.NaK_th = trn_pars.get('NaK_th')
  group.g_KL = group.suggest_gKL(rest_Vr, g_T=all_gT, Iext=0.)
  group.g_T = all_gT
  group.reset(rest_Vr)

  INa, IK, IT = group.get_channel_currents(V=init_Vr, y=init_Vr, z=init_Vr, g_T=all_gT)
  IKL = group.g_KL * (init_Vr - group.E_KL)
  IL = group.g_L * (init_Vr - group.E_L)
  Iext = (INa + IK + IT + IL + IKL) / group.V_factor

  # run neuron group
  group.reset(init_Vr)
  inputs = bp.inputs.section_input(values=[Iext, 0.],
                                   durations=[duration / 2, duration / 2],
                                   dt=bm.get_dt())

  runner = bp.DSRunner(group, inputs=['input', inputs, 'iter'], monitors=['V', 'spike'])
  runner(duration=duration)
  spike_num = runner.mon.spike.sum(axis=0)

  if fn:
    fn = f'{fn}/rand={method}-size={size[0]}-{size[1]},init={init_Vr},Vr={rest_Vr},gT={gT},sigma={gT_sigma}'
    if not os.path.exists(fn):
      os.makedirs(fn)

    np.save(f'{fn}-data.npy', spike_num)
    plt.hist(spike_num)
    plt.savefig(f'{fn}-overall.png')


def visualize_results():
  data_fn = 'Fig2/rand=gaussian-size=30-30,init=-75.0,Vr=-68.0,gT=3.25,sigma=1.0-data.npy'
  data = np.load(data_fn)
  data = pd.DataFrame(data, columns=['spike_num'])

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])
  sns.histplot(data=data, x="spike_num", kde=True, ax=ax, stat='count', color="#6195C8")
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.xlabel('Spike number', fontsize=30)
  plt.ylabel('Count', fontsize=30)
  ax.spines['bottom'].set_linewidth(2)  
  ax.spines['left'].set_linewidth(2)    
  plt.tick_params(axis='both', which='major', width=2)  
  plt.tick_params(axis='both', which='minor', width=2)  
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  
  
  
  
  plt.savefig('Fig3/burst-spike-num-distribution.png', transparent=True, dpi=500)
  # plt.show()

# Fig2-E
def visualize_results():
  data = np.load('./data/Fig1/cc_vs_J.npz')
  all_cc = data['all_cc']
  all_cc = all_cc[~np.all(all_cc == 1., axis=1)]

  all_J = np.repeat(data['all_J'], repeats=all_cc.shape[0])
  data = np.asarray([all_cc.T.flatten(), all_J.flatten()]).T
  data = pd.DataFrame(data, columns=['CC', 'J'])

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])
  sns.lineplot(x='J', y='CC', data=data, ax=ax, linewidth=3)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  
  ax.spines['bottom'].set_linewidth(2)  
  ax.spines['left'].set_linewidth(2)    
  plt.tick_params(axis='both', which='major', width=2)  
  plt.tick_params(axis='both', which='minor', width=2)  
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  
  
  plt.ylabel('CC', fontsize=30)
  plt.xlabel('$J_e$', fontsize=30)
  
  plt.savefig('Fig3/CC_vs_J.png', transparent=True, dpi=500)
  plt.show()




# Fig1 - F
def evaluate_gKL_by_Vr(Vr, g_T=2.25):
  P = bp.tools.DotDict(g_Na=100., g_K=10., b=0.5, rho_p=0., IT_th=-3.,
                       NaK_th=-55., E_KL=-100, g_L=0.06, E_L=-70)
  trn = ReducedTRNv1(1)
  trn.g_Na = P['g_Na']
  trn.g_K = P['g_K']
  trn.b = P['b']
  trn.rho_p = P['rho_p']
  trn.IT_th = P['IT_th']
  trn.NaK_th = P['NaK_th']
  trn.E_KL = P['E_KL']
  trn.g_L = P['g_L']
  trn.E_L = P['E_L']
  trn.g_T = g_T
  trn.g_KL = trn.suggest_gKL(Vr=Vr, Iext=0., g_T=g_T)
  P['g_KL'] = trn.g_KL
  return trn.g_KL

def relation_of_gKL_Vr():
  all_vr = bm.arange(-85., -62, 0.1)
  all_gkl = evaluate_gKL_by_Vr(all_vr)

  plt.rcParams.update({"font.size": 15})
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(bm.as_numpy(all_gkl), bm.as_numpy(all_vr), linewidth=3)
  plt.ylabel('Vr [mV]', fontsize=30)
  plt.xlabel(r'$g_{KL} [\mathrm{mS/cm^2}]$', fontsize=30)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  
  ax.spines['bottom'].set_linewidth(2)  
  ax.spines['left'].set_linewidth(2)    
  plt.tick_params(axis='both', which='major', width=2)  
  plt.tick_params(axis='both', which='minor', width=2)  
  plt.xticks(fontsize=20)  
  plt.yticks(fontsize=20)  
  
  plt.savefig('./Fig3/gKL_Vr_relation.png', transparent=True, dpi=500)
  plt.show()
  
# Fig1-G
def gKL_background():
    fig, gs = bp.visualize.get_figure(1, 1, 1.2, 6.)
    ax = fig.add_subplot(gs[0, 0])
    plt.subplots_adjust(bottom=0.15, top=0.85)  
    plt.xlabel('$g_{KL}$', fontsize=30)
    plt.ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 0.06)
    plt.ylim(3, 8)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    plt.yticks([])  # 移除纵坐标刻度标签
    plt.tick_params(axis='both', which='major', width=0)  
    plt.xticks(fontsize=20)  
    
    x1 = np.linspace(0, 0.008, 20)
    y1 = np.ones(20) * 3
    y2 = np.ones(20) * 8
    plt.fill_between(x1, y1, y2, alpha=.5, linewidth=0, color='#9AC9DB')
    
    x2 = np.linspace(0.008, 0.022, 20)
    y3 = np.ones(20) * 3
    y4 = np.ones(20) * 8
    plt.fill_between(x2, y3, y4, alpha=.5, linewidth=0, color="#C82423")
    
    x3 = np.linspace(0.022, 0.06, 20)
    y5 = np.ones(20) * 3
    y6 = np.ones(20) * 8
    plt.fill_between(x3, y5, y6, alpha=.5, linewidth=0, color='#9AC9DB')

    #plt.tight_layout()
    plt.savefig("./Fig3/Figure2-SI_gKL_background.png", transparent=True, dpi=500)

def gKL_firingPattern(g_Na=100., g_K=10., b=0.5, rho_p=0., IT_th=-3.,
                     NaK_th=-55., E_KL=-100, g_L=0.06, E_L=-70,
                              Vr= -60., g_T=2.25, Iexts=0):    
    reduced = neus_and_syns.ReducedTRNv1(size=1, method='exp_auto')

    sns.set(font_scale=1.5)
    sns.set_style("white")

    reduced.NaK_th  = NaK_th
    reduced.g_T  = g_T
    reduced.g_L  = g_L
    reduced.E_L  = E_L
    reduced.E_KL = E_KL
    reduced.b = b
    reduced.rho_p = rho_p
    reduced.g_Na = g_Na
    reduced.g_K = g_K
    reduced.IT_th  = IT_th
    
    
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 4.)

    inputs, duration = bp.inputs.section_input(values=[0., Iexts, 0.], durations=[50., 100., 350.], return_length=True)

    reduced.reset(Vr=-59)
    reduced.g_KL = reduced.suggest_gKL(Vr=Vr, Iext=0., g_T=g_T)
    runnerReduced = bp.DSRunner(reduced, inputs=['input', inputs, 'iter'], monitors=['V', 'spike'])
    runnerReduced(duration=duration)

    start, end = 0, duration
    ax = fig.add_subplot(gs[0, 0])
    sns.set_palette("hls")  
    plt.plot(runnerReduced.mon.ts, runnerReduced.mon.V[:, 0], label='reduced', color="#6195C8", linewidth=4)
    plt.xlim(start, end)
    #plt.ylim(ax1_ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-80, 80)
    plt.tick_params(axis='both', which='major', width=3)  
    plt.xticks(fontsize=15)  
    plt.yticks(fontsize=15)  
    plt.xlabel('Time(ms)')  

    plt.savefig("./Fig1/Figure1_Vr_wxample3.png", transparent=True, dpi=500)  
    
    
if __name__ == '__main__':

  visualize_results()