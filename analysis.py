# -*- coding: utf-8 -*-

import os.path

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
  'TrajectModel',
  'PhasePlane',
  'FastSlowBifurcation',
]


class TrajectModel(bp.DynamicalSystem):
  def __init__(self, integral, initial_vars, pars=None, dt=None):
    super(TrajectModel, self).__init__()

    # variables
    assert isinstance(initial_vars, dict)
    initial_vars = {k: bm.Variable(bm.asarray(v)) for k, v in initial_vars.items()}
    self.register_implicit_vars(initial_vars)
    self.all_vars = tuple(self.implicit_vars.values())

    # parameters
    pars = dict() if pars is None else pars
    assert isinstance(pars, dict)
    self.pars = [bm.asarray(v) for k, v in pars.items()]

    # integrals
    self.integral = integral

    # runner
    self.runner = bp.DSRunner(self,
                              monitors=list(initial_vars.keys()),
                              dyn_vars=self.vars().unique(), dt=dt)

  def update(self, tdi):
    new_vars = self.integral(*self.all_vars, tdi.t, *self.pars, dt=tdi.dt)
    for var, value in zip(self.all_vars, new_vars):
      var.update(value)

  def __getattr__(self, item):
    child_vars = super(TrajectModel, self).__getattribute__('implicit_vars')
    if item in child_vars:
      return child_vars[item]
    else:
      return super(TrajectModel, self).__getattribute__(item)

  def run(self, duration):
    self.runner.run(duration)
    return self.runner.mon


class PhasePlane(bp.analysis.PhasePlane2D):
  @property
  def integral(self):
    def derivative(x, y, t, *pars):
      dx = self.F_fx(x, y, *pars)
      dy = self.F_fy(x, y, *pars)
      return dx, dy

    return bp.odeint(method='rk4', f=derivative)

  def plot_trajectory(self, initials, duration, plot_durations=None,
                      axes='v-v', dt=None, show=False, with_plot=True, with_return=False):
    assert axes == 'v-v'
    bp.analysis.utils.output('I am plotting trajectory ...')

    # check the initial values
    initials = bp.analysis.utils.check_initials(initials, self.target_var_names)

    # 2. format the running duration
    assert isinstance(duration, (int, float))

    # 3. format the plot duration
    plot_durations = bp.analysis.utils.check_plot_durations(plot_durations, duration, initials)

    # 5. run the network
    dt = bm.get_dt() if dt is None else dt
    traject_model = TrajectModel(initial_vars=initials,
                                 integral=self.integral,
                                 dt=dt)
    mon_res = traject_model.run(duration=duration)

    if with_plot:
      # plots
      for i, initial in enumerate(zip(*list(initials.values()))):
        # legend
        legend = f'$traj_{i}$: '
        for j, key in enumerate(self.target_var_names):
          legend += f'{key}={initial[j]}, '
        legend = legend[:-2]

        # visualization
        start = int(plot_durations[i][0] / dt)
        end = int(plot_durations[i][1] / dt)
        if axes == 'v-v':
          lines = plt.plot(mon_res[self.x_var][start: end, i], mon_res[self.y_var][start: end, i], label=legend)
          bp.analysis.utils.add_arrow(lines[0])
        else:
          plt.plot(mon_res.ts[start: end], mon_res[self.x_var][start: end, i],
                   label=legend + f', {self.x_var}')
          plt.plot(mon_res.ts[start: end], mon_res[self.y_var][start: end, i],
                   label=legend + f', {self.y_var}')

      # visualization of others
      if axes == 'v-v':
        plt.xlabel(self.x_var)
        plt.ylabel(self.y_var)
        scale = (self.lim_scale - 1.) / 2
        plt.xlim(*bp.analysis.utils.rescale(self.target_vars[self.x_var], scale=scale))
        plt.ylim(*bp.analysis.utils.rescale(self.target_vars[self.y_var], scale=scale))
        plt.legend()
      else:
        plt.legend(title='Initial values')

      if show:
        plt.show()

    if with_return:
      return mon_res

  def plot_limit_cycle_by_sim(self, initials, duration, tol=0.001, show=False, dt=None):
    raise NotImplementedError


class FastSlowBifurcation(bp.analysis.FastSlow2D):
  @property
  def integral(self):
    def derivative(x, y, t, *pars):
      dx = self.F_fx(x, y, *pars)
      dy = self.F_fy(x, y, *pars)
      return dx, dy

    return bp.odeint(method='rk4', f=derivative)

  def plot_limit_cycle_by_sim(self, duration=100, with_plot=True, with_return=False,
                              plot_style=None, tol=0.001, show=False, dt=None, offset=1., I_syn=0., gT=2.0,
                              real_Vr=-70., save_path='./'):
    bp.analysis.utils.output('I am plotting limit cycle ...')

    if self._fixed_points is None:
      bp.analysis.utils.output('No fixed points found, you may call "plot_bifurcation(with_plot=True)" first.')
      return

    final_fps, final_pars = self._fixed_points
    dt = bm.get_dt() if dt is None else dt
    traject_model = TrajectModel(
      initial_vars={self.x_var: final_fps[:, 0] + offset,
                    self.y_var: final_fps[:, 1] + offset},
      integral=self.integral,
      pars={p: v for p, v in zip(self.target_par_names, final_pars.T)},
      dt=dt
    )
    mon_res = traject_model.run(duration=duration)

    # find limit cycles
    vs_limit_cycle = tuple({'min': [], 'max': []} for _ in self.target_var_names)
    ps_limit_cycle = tuple([] for _ in self.target_par_names)
    for i in range(mon_res[self.x_var].shape[1]):
      data = mon_res[self.x_var][:, i]
      max_index = bp.analysis.utils.find_indexes_of_limit_cycle_max(data, tol=tol)
      if max_index[0] != -1:
        cycle = data[max_index[0]: max_index[1]]
        vs_limit_cycle[0]['max'].append(mon_res[self.x_var][max_index[1], i])
        vs_limit_cycle[0]['min'].append(cycle.min())
        cycle = mon_res[self.y_var][max_index[0]: max_index[1], i]
        vs_limit_cycle[1]['max'].append(mon_res[self.y_var][max_index[1], i])
        vs_limit_cycle[1]['min'].append(cycle.min())
        for j in range(len(self.target_par_names)):
          ps_limit_cycle[j].append(final_pars[i, j])
    vs_limit_cycle = tuple({k: np.asarray(v) for k, v in lm.items()} for lm in vs_limit_cycle)
    ps_limit_cycle = tuple(np.array(p) for p in ps_limit_cycle)

    # visualization
    if with_plot:
      if plot_style is None: plot_style = dict()
      fmt = plot_style.pop('fmt', '.')

      if len(self.target_par_names) == 2:
        for i, var in enumerate(self.target_var_names):
          plt.figure(var)
          plt.plot(ps_limit_cycle[0], ps_limit_cycle[1], vs_limit_cycle[i]['max'],
                   **plot_style, label='limit cycle (max)')
          plt.plot(ps_limit_cycle[0], ps_limit_cycle[1], vs_limit_cycle[i]['min'],
                   **plot_style, label='limit cycle (min)')
          plt.legend()
          # plt.title(f'gT={gT}')

          if save_path:
            if not os.path.exists(save_path):
              os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'FSB_{}_{}_{}_{}_{}.png'.format(i, var, I_syn, gT, real_Vr)))
            plt.savefig(os.path.join(save_path, 'FSB_{}_{}_{}_{}_{}.eps'.format(i, var, I_syn, gT, real_Vr)))
          else:
            plt.show()
          plt.close()


      elif len(self.target_par_names) == 1:
        for i, var in enumerate(self.target_var_names):
          plt.figure(var)
          plt.plot(ps_limit_cycle[0], vs_limit_cycle[i]['max'], fmt,
                   **plot_style, label='limit cycle (max)')
          plt.plot(ps_limit_cycle[0], vs_limit_cycle[i]['min'], fmt,
                   **plot_style, label='limit cycle (min)')
          plt.legend()
          # plt.title(f'gT={gT}')

          if save_path:
            if not os.path.exists(save_path):
              os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'FSB_{}_{}_{}_{}_{}.png'.format(i, var, I_syn, gT, real_Vr)))
            plt.savefig(os.path.join(save_path, 'FSB_{}_{}_{}_{}_{}.eps'.format(i, var, I_syn, gT, real_Vr)))
          else:
            plt.show()
          plt.close()

      else:
        raise bp.errors.AnalyzerError

      if show:
        plt.show()

    if with_return:
      return vs_limit_cycle, ps_limit_cycle
