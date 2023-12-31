# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:00:05 2023

@author: Shangyang
"""


import os.path

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

import neus_and_syns
import utils

bp.math.enable_x64()
bp.math.set_platform('cpu')
bp.math.set_dt(0.01)


def run_net(
    # parameters for network components
    trn_pars, poisson_pars, gaba_pars, gj_pars, conn_pars, 
    # key parameters of TRN neurons
    Vr_and_gT, all_gT, input1,
    ampa_pars=None,
    # simulation settings
    run_duration=1e4, run_step=1.5e4, report=False, method='rk4', size=(10, 10),
    # output settings
    name_pars=None, save_path=None, save_lfp=False, data_save = False,
):
  # connection parameters
  assert isinstance(conn_pars, dict)
  if conn_pars['method'] == 'grid_four':
    conn = bp.conn.GridFour()(size)
  elif conn_pars['method'] == 'grid_eight':
    conn = bp.conn.GridEight()(size)
  elif conn_pars['method'] == 'gaussian_prob':
    conn = neus_and_syns.GaussianProbForGJ(**conn_pars.get('pars', dict()))
  else:
    raise ValueError

  # network
  if ampa_pars is None:
    net = neus_and_syns.TRNNet(size=size, conn=conn, method=method)
  else:
    net = neus_and_syns.TRNNet_ExternalAMPA(size=size, conn=conn, method=method, ampa_pars=ampa_pars)

  # TRN parameters
  assert isinstance(trn_pars, dict)
  for key, value in trn_pars.items():
    if hasattr(net.T, key):
      setattr(net.T, key, value)
    else:
      raise ValueError(f'{net.T} does not have "{key}".')

  # poisson parameters
  assert isinstance(poisson_pars, dict)
  for key, value in poisson_pars.items():
    if hasattr(net.P, key):
      setattr(net.P, key, value)
    else:
      raise ValueError(f'{net.P} does not have "{key}".')

  # GABAA parameters
  assert isinstance(gaba_pars, dict)
  for key, value in gaba_pars.items():
    if hasattr(net.P2T, key):
      setattr(net.P2T, key, value)
    else:
      raise ValueError(f'{net.P2T} does not have "{key}".')

  # gap junction parameters
  assert isinstance(gj_pars, dict)
  for key, value in gj_pars.items():
    if hasattr(net.GJ, key):
      setattr(net.GJ, key, value)
    else:
      raise ValueError(f'{net.GJ} does not have "{key}".')

  # initialize TRN group
  Vr, gT = Vr_and_gT
  gKL = net.T.suggest_gKL(Vr=Vr, g_T=gT, Iext=0.)
  gKL = bm.as_jax(bm.where(gKL < 0., 0., gKL))

  gjw = gj_pars['gjw']
  if report:
    print(f'gap junction weight = {gjw}')
    print(f'For neurons with gT={gT}, we suggest gKL={gKL} when Vr={Vr} mV.')

  net.T.g_KL = gKL
  net.T.g_T = bm.asarray(all_gT)
  net.reset((bm.random.random(size=net.T.num) - 0.5) * 10 + Vr)

  # run the network
  inpys = bp.inputs.section_input([input1, input1, input1], [7000., 1000., 7000])

  runner = bp.DSRunner(net, inputs=[('T.input', inpys, 'iter')],
                       monitors=['T.spike', 'T.V'],
                       dyn_vars=net.vars().unique())


  # output parameter
  name_pars = dict() if name_pars is None else name_pars
  name_pars = ','.join([f'{k}={v}' for k, v in name_pars.items()])

  potentials = []
  spikes = []

  for i, start in enumerate(np.arange(0, run_duration, run_step)):
    end = min(run_duration, start + run_step)
    t = runner.run(end - start)

    potentials.append(runner.mon['T.V'])
    spikes.append(runner.mon['T.spike'])

    # raster plot
    fig, gs = bp.visualize.get_figure(row_num=5, col_num=1, row_len=2, col_len=20)
    ax = fig.add_subplot(gs[0:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['T.spike'], xlim=(start, end), xlabel=None, ylabel=None)
    plt.tick_params(labelsize=20)
    ax.get_xaxis().set_ticks([])
    title = name_pars + f',run={i}' if name_pars else f'run={i}'
    #plt.title(title)
    
    
    # ax = fig.add_subplot(gs[2, 0])
    # bp.visualize.line_plot(runner.mon.ts, inpys,
    #                        xlim=(start, end), ylabel=None, xlabel=None)
    # plt.tick_params(labelsize=20)
    # ax.get_xaxis().set_ticks([])
    

    # membrane potential
    ax = fig.add_subplot(gs[2, 0])
    bp.visualize.line_plot(runner.mon.ts, runner.mon['T.V'],
                           plot_ids=np.random.randint(0, bp.tools.size2num(size), 3),
                           xlim=(start, end), ylabel=None, xlabel=None, legend='N')
    # plt.ylabel('V',fontsize=30)
    plt.tick_params(labelsize=20)
    ax.get_xaxis().set_ticks([])
    plt.legend(loc='upper right', fontsize=15)

    # LFP
    ax = fig.add_subplot(gs[3, 0])
    fs = 1e3 / bm.get_dt()
    window_sec = 2 / 0.5
    s_time = int(150 / bm.get_dt()) if i == 0 else 0
    lfp = np.mean(runner.mon['T.V'], axis=1)
    mean_lfp = lfp[s_time:] - np.mean(lfp[s_time:])
    filtered_lfp = utils.bandpass(mean_lfp, 1, 30, fs, corners=3)
    lfp_freqs, psd = utils.bandpower(filtered_lfp, fs, window_sec)

    plt.plot(runner.mon.ts[s_time:], mean_lfp, 'k', label='Raw TRN')
    plt.plot(runner.mon.ts[s_time:], filtered_lfp, 'r', label='Filtered TRN')
    plt.xlim((start, end))
    plt.legend(loc='upper right', fontsize=15)
    # plt.ylabel('Freq={:.3f}'.format(freq),fontsize=30)
    plt.tick_params(labelsize=20)
    ax.get_xaxis().set_ticks([])
    # plt.xlabel('Time [ms]')

    # firing rate
    fig.add_subplot(gs[4, 0])
    pop_firing = bp.measure.firing_rate(runner.mon['T.spike'], width=10)

    plt.plot(runner.mon.ts[s_time:], pop_firing[s_time:], 'k', label='firing tate')
    plt.xlim((start, end))
    # plt.legend(loc='upper right')
    plt.xlabel('Time [ms]', fontsize=30)
    plt.tick_params(labelsize=20)
    # plt.legend()

    # # power
    # length = int(30 * window_sec)
    # fig.add_subplot(gs[6, 0])
    # plt.plot(lfp_freqs[:length], psd[:length])
    # plt.xlabel('Frequency [Hz]', fontsize=30)
    # # plt.ylabel('Power={:.3f}'.format(power),fontsize=30)
    # plt.tick_params(labelsize=20)

    if save_path:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, title + '.png'), dpi=500, transparent=False)

      if save_lfp:
        np.savez(os.path.join(save_path, title), lfp=lfp)
    else:
      plt.show()
    plt.close(fig)

  # save Local Field Potential (LFP)
  if data_save == True:
      if save_path:
        if not os.path.exists(save_path):
          os.makedirs(save_path)
        np.savez(file=os.path.join(save_path, title[:-6] + '.npz'),
                 potentials=np.concatenate(potentials, axis=0),
                 spikes=np.concatenate(spikes, axis=0))


if __name__ == '__main__':
  # -- set pars -- #
  size_ = (10, 10)
  # trn pars
  b = 0.5
  rho_p = 0.01
  IT_th = -3.
  NaK_th = -50.
  E_KL = -100.
  g_L = 0.05
  E_L = -60.
  # poisson pars
  freqs = 0
  # gaba pars
  gaba_pars = dict(alpha=10., E=-50., g_max=0.001) #0.001
  #gaba_pars = dict(alpha=10., E=-60., g_max=0.01)
  # connection type
  conn_pars = dict(method='gaussian_prob', pars=dict(sigma=0.7, periodic_boundary=True, seed=32))
  #ampa_pars = dict(g_max=0.01)
  ampa_pars = None
  gT = 0
  Vr = -64.1  #-60
  gT_sigma = 0
  gjw = 0.001
  aa = 0.12
  print(f'gT_sigma = {gT_sigma}')
  print(f'Vr = {Vr}')
  run_net(
    trn_pars=dict(b=b, rho_p=rho_p, IT_th=IT_th, NaK_th=NaK_th, E_KL=E_KL, g_L=g_L, E_L=E_L),
    poisson_pars=dict(freqs=freqs),
    gaba_pars=gaba_pars,
    ampa_pars=ampa_pars,
    gj_pars=dict(gjw=gjw),
    conn_pars=conn_pars,
    input1 = aa,
    name_pars=dict(input1=aa, freqs=freqs, gT=gT, gT_sigma=gT_sigma, Vr=Vr, gjw=gjw,
                   b=b, rho_p=rho_p, E_KL=E_KL, g_L=g_L, E_L=E_L,
                   gaba_g_max=gaba_pars['g_max'], gaba_E=gaba_pars['E'],
                   amap_g_max='None' if ampa_pars is None else ampa_pars['g_max']),
    Vr_and_gT=[Vr, gT],
    all_gT=utils.uniform1(size=bp.tools.size2num(size_), ranges=(gT - gT_sigma, gT + gT_sigma), seed=57362),
    size=size_,
    run_duration=1.5e4,
    run_step=1.5e4,
    method='rk4',
    report=True,
    save_path=f'./Fig1207/gT=0-gT_sigmas={gT_sigma}/gjw={gjw}/'
  )