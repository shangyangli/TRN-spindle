# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import neus_and_syns


#bp.math.enable_x64(True)
bp.math.set_platform('cpu')
bp.math.set_dt(0.01)


def run_net(
    # parameters for network components
    trn_pars, poisson_pars, gaba_pars, gj_pars, conn_pars,
    # key parameters of TRN neurons
    Vr_and_gT, all_gT,
    # simulation settings
    report=False, method='rk4', size=(10, 10), monitors=('T.spike', 'T.V'), duration=1e3,
    rng=bm.random, numpy=True, progress_bar=True,
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
  net.reset((rng.random(size=net.T.num) - 0.5) * 10 + Vr)

  # run the network
  runner = bp.DSRunner(net, monitors=monitors, numpy_mon_after_run=numpy, progress_bar=progress_bar)
  runner.run(duration)

  return runner.mon


