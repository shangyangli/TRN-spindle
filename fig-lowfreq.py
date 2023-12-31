# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:13:11 2023

@author: Shangyang
"""

import os.path

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

import neus_and_syns
import spindle_detection
import utils

bp.math.enable_x64(True)
bp.math.set_platform('cpu')
bp.math.set_dt(0.01)


def run_net(
    # parameters for network components
    gT_sigma, trn_pars, poisson_pars, gaba_pars, gj_pars, conn_pars,
    # key parameters of TRN neurons
    Vr_and_gT, all_gT,
    # simulation settings
    run_duration=1e4, run_step=1.5e4, report=False, method='rk4', size=(10, 10),
    # output settings
    name_pars=None, save_path=None, save_lfp=False,
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
  net = neus_and_syns.TRNNet(size=size, conn=conn, method=method)

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

  gjw = gj_pars['gjw']
  if report:
    print(f'gap junction weight = {gjw}')
    print(f'For neurons with gT={gT}, we suggest gKL={gKL} when Vr={Vr} mV.')

  net.T.g_KL = gKL
  net.T.g_T = bm.asarray(all_gT)
  net.reset((bm.random.random(size=net.T.num) - 0.5) * 10 + Vr)

  # run the network
  runner = bp.DSRunner(net,
                       monitors=['T.spike', 'T.V'],
                       dyn_vars=net.vars().unique())

  # output parameter
  name_pars = dict() if name_pars is None else name_pars
  name_pars = ','.join([f'{k}={v}' for k, v in name_pars.items()])

  lfps = []

  for i, start in enumerate(np.arange(0, run_duration, run_step)):
    end = min(run_duration, start + run_step)
    t = runner.run(end - start)

    # raster plot
    fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, row_len=2, col_len=20)
    ax = fig.add_subplot(gs[0:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['T.spike'], xlim=(start, end), xlabel=None, ylabel=None)
    plt.ylabel('Neuron Index', fontsize=20)
    plt.tick_params(labelsize=20)
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    title = name_pars + f',run={i}' if name_pars else f'run={i}'
    # plt.title(title)
    plt.xlabel('Time [ms]', fontsize=30)

    if save_path:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, title + 'raster.png'))
      plt.savefig(os.path.join(save_path, title + 'raster.eps'))
    else:
      plt.show()
    plt.close(fig)

    # membrane potential
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=2, col_len=20)
    ax = fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(runner.mon.ts, runner.mon['T.V'],
                           plot_ids=np.random.randint(0, bp.tools.size2num(size), 3),
                           xlim=(start, end), ylabel=None, xlabel=None, legend='N')
    plt.ylabel('V', fontsize=30)
    plt.tick_params(labelsize=20)
    ax.get_xaxis().set_ticks([])
    plt.legend(loc='upper right', fontsize=15)
    plt.xlabel('Time [ms]', fontsize=30)

    if save_path:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, title + 'example.png'))
      plt.savefig(os.path.join(save_path, title + 'example.eps'))
    else:
      plt.show()
    plt.close(fig)

    # LFP
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=2, col_len=20)
    ax = fig.add_subplot(gs[0, 0])
    fs = 1e3 / bm.get_dt()
    window_sec = 2 / 0.5
    s_time = int(150 / bm.get_dt()) if i == 0 else 0
    lfp = np.mean(runner.mon['T.V'], axis=1)
    mean_lfp = lfp[s_time:] - np.mean(lfp[s_time:])
    filtered_lfp = utils.bandpass(mean_lfp, 1, 30, fs, corners=3)
    lfp_freqs, psd = utils.bandpower(filtered_lfp, fs, window_sec)
    idx = np.argmax(psd)
    freq = lfp_freqs[idx]
    power = psd[idx]
    lfps.append(mean_lfp)  # 选取原始LFP作为spindle detect？

    plt.plot(runner.mon.ts[s_time:], mean_lfp, 'k', label='Raw TRN')
    plt.plot(runner.mon.ts[s_time:], filtered_lfp, 'r', label='Filtered TRN')
    plt.xlim((start, end))
    plt.legend(loc='upper right', fontsize=15)
    # plt.ylabel('Freq={:.3f}'.format(freq),fontsize=30)
    plt.tick_params(labelsize=20)
    ax.get_xaxis().set_ticks([])
    plt.xlabel('Time [ms]', fontsize=30)

    if save_path:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, title + 'LFP.png'))
      plt.savefig(os.path.join(save_path, title + 'LFP.eps'))
    else:
      plt.show()
    plt.close(fig)

    # firing rate
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=2, col_len=20)
    fig.add_subplot(gs[0, 0])
    pop_firing = bp.measure.firing_rate(runner.mon['T.spike'], width=10)

    plt.plot(runner.mon.ts[s_time:], pop_firing[s_time:], 'k', label='firing tate')
    plt.xlim((start, end))
    # plt.legend(loc='upper right')
    plt.xlabel('Time [ms]', fontsize=30)
    plt.tick_params(labelsize=20)
    # plt.legend()

    if save_path:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, title + 'firing.png'))
      plt.savefig(os.path.join(save_path, title + 'firing.eps'))
    else:
      plt.show()
    plt.close(fig)

    # power
    fig, gs = bp.visualize.get_figure(row_num=1, col_num=1, row_len=2, col_len=20)
    length = int(30 * window_sec)
    fig.add_subplot(gs[0, 0])
    plt.plot(lfp_freqs[:length], psd[:length])
    plt.xlabel('Frequency [Hz]', fontsize=30)
    # plt.ylabel('Power={:.3f}'.format(power),fontsize=30)
    plt.tick_params(labelsize=20)

    # plt.tight_layout()

    if save_path:
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      plt.savefig(os.path.join(save_path, title + 'power.png'))
      plt.savefig(os.path.join(save_path, title + 'power.eps'))

      if save_lfp:
        np.savez(os.path.join(save_path, title), lfp=lfp)
    else:
      plt.show()
    plt.close(fig)

  # save Local Field Potential (LFP)
  if save_path:
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    postfix = f'gjw={gjw},Vr={Vr},gT_sigma = {gT_sigma}'  ####
    filename = save_path + f'/{postfix}.npz'
    print(filename, '\n\n')
    np.savez(file=filename, lfp=np.concatenate(lfps))


# --- Helpers of spindle detection --- #
def spindle_detect(file_name,
                   return_res,
                   freq_sp=(12, 15),
                   freq_broad=(1, 30),
                   min_distance=250,
                   thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                   corr_window=0.1,
                   corr_step=0.05,
                   rms_cut=0.025,
                   soft_width=0.1,
                   sp_threshold=2.01,
                   gjw=0.01):
  print(file_name)
  lfp = np.load(file_name)['lfp']
  sps = spindle_detection.spindles_detect(
    data=lfp, sf=1e3 / 0.01, thresh=thresh, freq_sp=freq_sp, freq_broad=freq_broad,
    min_distance=min_distance, corr_window=corr_window, corr_step=corr_step, rms_cut=rms_cut,
    soft_width=soft_width, sp_threshold=sp_threshold,
  )
  obj = dir(sps)
  if 'summary' in obj:
    return_res[file_name] = sps.summary()
  else:
    print(f'No spindle when giw={gjw}')
    
# ---- Low frequency oscillation example ---- #
if __name__ == '__main__1':
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
  freqs = 50
  # gaba pars
  alpha = 10.
  E = -60.  # -62.
  g_max = 0.0004
  # connection type
  sigma = 0.7
  conn_pars = dict(method='gaussian_prob', pars=dict(sigma=sigma, periodic_boundary=True, seed=32))
  # connection visualization
  visualiztion = False
  # gap junction weight
  gjws = [0.001]  
  Vrs = [-73]  
  gT_sigmas = [0.8]
  gT = 2.6

  for Vr in Vrs:
    for gjw in gjws:
      for gT_sigma in gT_sigmas:
        print(f'gT_sigma = {gT_sigma}')
        print(f'Vr = {Vr}')
        run_net(
          gT_sigma=gT_sigma,
          trn_pars=dict(b=b, rho_p=rho_p, IT_th=IT_th, NaK_th=NaK_th,
                        E_KL=E_KL, g_L=g_L, E_L=E_L),
          poisson_pars=dict(freqs=freqs),
          gaba_pars=dict(alpha=alpha, E=E, g_max=g_max),
          gj_pars=dict(gjw=gjw),
          conn_pars=conn_pars,
          # conn_pars=dict(method='grid_four'),
          name_pars=dict(gT_sigma=gT_sigma, Vr=Vr, gjw=gjw, freqs=freqs, b=b, rho_p=rho_p, E_KL=E_KL, g_L=g_L, E_L=E_L,
                         g_max=g_max),
          Vr_and_gT=[Vr, 2.5],
          all_gT=utils.uniform1(size=bp.tools.size2num(size_), ranges=(gT - gT_sigma, gT + gT_sigma), seed=57362),
          size=size_,
          run_duration=6e4,
          run_step=1.5e4,
          method='rk4',
          report=True,
          save_path=f'./data/Fig3_6/Vr={Vr}/gjw={gjw}/'
        )
