# -*- coding: utf-8 -*-

import os
import math
from typing import Sequence, Dict, Union

import brainpy as bp
import matplotlib.pyplot as plt
import numba
import numpy as np
from jax import vmap, pmap
from jax.tree_util import tree_unflatten, tree_flatten
from scipy.integrate import simps
from scipy.signal import iirfilter
from scipy.signal import sosfilt
from scipy.signal import welch
from scipy.signal import zpk2sos


@numba.njit
def identity(a, b, tol=0.01):
  if np.abs(a - b) < tol:
    return True
  else:
    return False


def find_limit_cycle_max(arr, tol=0.01):
  if np.ndim(arr) == 1:
    grad = np.gradient(arr)
    condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] >= 0)
    indexes = np.where(condition)[0]
    if len(indexes) >= 2:
      if identity(arr[indexes[-2]], arr[indexes[-1]], tol):
        return indexes[-1]
    return -1

  elif np.ndim(arr) == 2:
    # The data with the shape of (axis_along_time, axis_along_neuron)
    grads = np.gradient(arr, axis=0)
    conditions = np.logical_and(grads[:-1] * grads[1:] <= 0, grads[:-1] >= 0)
    indexes = -np.ones(len(conditions), dtype=int)
    for i, condition in enumerate(conditions):
      idx = np.where(condition)[0]
      if len(idx) >= 2:
        if identity(arr[idx[-2]], arr[idx[-1]], tol):
          indexes[i] = idx[-1]
    return indexes

  else:
    raise ValueError


def find_limit_cycle_min(arr, tol=0.01):
  if np.ndim(arr) == 1:
    grad = np.gradient(arr)
    condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] <= 0)
    indexes = np.where(condition)[0]
    if len(indexes) >= 2:
      indexes += 1
      if identity(arr[indexes[-2]], arr[indexes[-1]], tol):
        return indexes[-1]
    return -1

  elif np.ndim(arr) == 2:
    # The data with the shape of (axis_along_time, axis_along_neuron)
    grads = np.gradient(arr, axis=0)
    conditions = np.logical_and(grads[:-1] * grads[1:] <= 0, grads[:-1] <= 0)
    indexes = -np.ones(len(conditions), dtype=int)
    for i, condition in enumerate(conditions):
      idx = np.where(condition)[0]
      if len(idx) >= 2:
        idx += 1
        if identity(arr[idx[-2]], arr[idx[-1]], tol):
          indexes[i] = idx[-1]
    return indexes

  else:
    raise ValueError


def find_local_maxima(arr):
  if np.ndim(arr) == 1:
    grad = np.gradient(arr)
    condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] >= 0)
    indexes = np.where(condition)[0]
    return indexes + 1
  elif np.ndim(arr) == 2:
    # The data with the shape of (axis_along_time, axis_along_neuron)
    grads = np.gradient(arr, axis=0)
    conditions = np.logical_and(grads[:-1] * grads[1:] <= 0, grads[:-1] >= 0)
    indexes = np.where(conditions)
    return indexes[0] + 1, indexes[1] + 1
  else:
    raise ValueError


def find_local_minima(arr):
  if np.ndim(arr) == 1:
    grad = np.gradient(arr)
    condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] <= 0)
    indexes = np.where(condition)[0]
    return indexes + 1
  elif np.ndim(arr) == 2:
    # The data with the shape of (axis_along_time, axis_along_neuron)
    grads = np.gradient(arr, axis=0)
    conditions = np.logical_and(grads[:-1] * grads[1:] <= 0, grads[:-1] <= 0)
    indexes = np.where(conditions)
    return indexes[0] + 1, indexes[1] + 1
  else:
    raise ValueError


def bandpass(data, freq_min, freq_max, fs, corners=4):
  """Butterworth-Bandpass Filter.

  Filter data from ``freq_min`` to ``freq_max`` using ``corners``
  corners. The filter uses :func:`scipy.signal.iirfilter` (for design)
  and :func:`scipy.signal.sosfilt` (for applying the filter).

  Parameters
  ----------
  data : numpy.ndarray
      Data to filter.
  freq_min : int, float
      Pass band low corner frequency.
  freq_max : int, float
      Pass band high corner frequency.
  fs : int, float
      Sampling rate in Hz.
  corners : int
      Filter corners / order.

  Returns
  -------
  filtered_data : numpy.ndarray
      Filtered data.
  """
  fe = 0.5 * fs
  low = freq_min / fe
  high = freq_max / fe
  if high - 1.0 > -1e-6:
    msg = "Selected high corner frequency ({}) of bandpass is at " \
          "or above Nyquist ({}). Applying a high-pass " \
          "instead.".format(freq_max, fe)
    raise ValueError(msg)
  if low > 1:
    msg = "Selected low corner frequency is above Nyquist."
    raise ValueError(msg)
  z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
  sos = zpk2sos(z, p, k)
  return sosfilt(sos, data)


def bandpower(data, fs, window_sec, band=None):
  """Compute the average power of the signal x in a specific frequency band.

  Parameters
  ----------
  data : 1d-array
      Input signal in the time-domain.
  fs : float
      Sampling frequency of the data.
  window_sec : float
      Length of each window in seconds.
      If None, window_sec = (1 / min(band)) * 2
  band : list
      Lower and upper frequencies of the band of interest.

  Return
  ------
  bp : float
      Absolute or relative band power.
  """
  # Define window length
  nperseg = window_sec * fs

  # Compute the modified periodogram (Welch)
  freqs, psd = welch(data, fs, nperseg=nperseg)

  if band is not None:
    low, high = band

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Integral approximation of the spectrum using Simpson's rule.
    power = simps(psd[idx_band], dx=freq_res)
    total_power = simps(psd, dx=freq_res)

    return power, power / total_power
  else:
    return freqs, psd


def show_LFP_and_Power(times, potential, window_sec=2 / 0.5):
  fs = 1e3 / bp.backend.get_dt()
  length = int(30 * window_sec)
  fig, gs = bp.visualize.get_figure(2, 2, 4, 8)

  # plot simulated local field potential of TRN
  fig.add_subplot(gs[0, 0])
  lfp = np.mean(potential, axis=1)
  lfp -= np.mean(lfp)
  filtered_lfp = bandpass(lfp, 0.5, 30, fs, corners=3)
  plt.plot(times, lfp, 'k', label='Raw TRN')
  plt.plot(times, filtered_lfp, 'r', label='Filtered TRN')
  plt.legend(loc='upper right')

  # plot power spectrum
  fig.add_subplot(gs[0, 1])
  freqs, psd = bandpower(filtered_lfp, fs, window_sec)
  plt.plot(freqs[:length], psd[:length], )
  plt.ylim(-1, np.max([100, np.max(psd)]) + 1)
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('TRN Power')


# @numba.njit
def _detect_spindle(times, signals, spindle_range, spindle_interval):
  _times = numba.typed.List.empty_list(numba.types.float64)
  _signals = numba.typed.List.empty_list(numba.types.float64)

  pre_larger_than_interval = True
  still_in_spindle = True

  i = 0  # current time
  while i < len(times) - 1:
    # time difference between the
    # current and the last point
    need_judge_end = False
    time_diff = times[i + 1] - times[i]
    if time_diff < spindle_range[1]:
      if time_diff >= spindle_range[0]:
        _times.append(times[i])
        _signals.append(signals[i])
      else:
        if len(_times) > 0:
          if times[i + 1] - _times[-1] >= spindle_range[1]:
            _times.append(times[i])
            _signals.append(signals[i])
            # i += 1
            # if i < (len(times) - 1):
            #     time_diff = times[i + 1] - times[i - 1]
          else:
            if signals[i] < signals[i + 1]:
              _times.append(times[i + 1])
              _signals.append(signals[i + 1])
              i += 1
              if i < (len(times) - 1):
                time_diff = times[i + 1] - times[i]
            else:
              _times.append(times[i])
              _signals.append(signals[i])
              i += 1
              if i < (len(times) - 1):
                time_diff = times[i + 1] - times[i - 1]
          if time_diff > spindle_range[1]:
            need_judge_end = True
        else:
          if signals[i] < signals[i + 1]:
            _times.append(times[i + 1])
            _signals.append(signals[i + 1])
            i += 1
            if i < (len(times) - 1):
              time_diff = times[i + 1] - times[i]
          else:
            _times.append(times[i])
            _signals.append(signals[i])
            i += 1
            if i < (len(times) - 1):
              time_diff = times[i + 1] - times[i - 1]
          if time_diff > spindle_range[1]:
            need_judge_end = True
      still_in_spindle = True
    else:
      need_judge_end = True

    if need_judge_end:
      if still_in_spindle:
        if time_diff > spindle_interval:
          if times[i] - _times[-1] >= spindle_range[0]:
            _times.append(times[i])
            _signals.append(signals[i])
          if pre_larger_than_interval:
            yield [_times, _signals]
          pre_larger_than_interval = True
        else:
          pre_larger_than_interval = False
      _times = numba.typed.List.empty_list(numba.types.float64)
      _signals = numba.typed.List.empty_list(numba.types.float64)
      still_in_spindle = False
    i += 1
  else:
    if still_in_spindle:
      _times.append(times[i])
      _signals.append(signals[i])
      yield [_times, _signals]


def detect_spindle(signals,
                   threshold=2.,
                   need_filter=False,
                   t_start=0.,
                   spindle_range=(5, 14),
                   spindle_interval=250,
                   dt=0.01):
  #               ( min,                   max                  )
  spindle_range = np.array([1e3 / spindle_range[1], 1e3 / spindle_range[0]])

  # get the maximum inflection points
  if need_filter:
    signals = bandpass(signals, 0.5, 30, fs=1e3 / dt, corners=3)
  grad = np.gradient(signals)
  condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] >= 0)
  condition = np.logical_and(condition, signals[:-1] > threshold)
  indices = np.array(np.where(condition)[0]) + 1
  signals = signals[indices]
  times = indices * dt + t_start

  # get the spindles
  # spindles = list(_detect_spindle(times, signals, spindle_range, spindle_interval))

  spindles = []
  for res in _detect_spindle(times, signals, spindle_range, spindle_interval):
    spindles.append({'time': np.array(res[0]), 'signal': np.array(res[1])})
  return spindles


def show_grid_four(net_size, i, j, w,
                   size=2, max_arrow_width=0.03,
                   arrow_sep=0.02, alpha=0.5, show=False):
  def get_color_by_weight(w):
    if w < 0.00005:
      return 'gray'
    if w < 0.001:
      return 'b'
    return 'r'

  def get_position(d1, d2):
    # 1: is width
    # 0: is height
    if d1[1] - d2[1] == 0:
      if d1[0] - d2[0] > 0:
        return 'right'
      else:
        return 'left'
    elif d1[0] - d1[0] == 0:
      if d1[1] - d2[1] > 0:
        return 'bottom'
      else:
        return 'top'
    else:
      raise ValueError('Same point.')

  height, width = net_size
  fig, gs = bp.visualize.get_figure(1, 1, 1 * height, 1 * width)
  ax = fig.add_subplot(gs[0, 0])

  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)

  plt.axis([-0.5, width - 0.5, -0.5, height - 0.5])
  plt.xticks([])
  plt.yticks([])
  max_text_size = size * 12
  label_text_size = size * 4

  text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif', 'fontweight': 'bold'}
  for h in range(height):
    for c in range(width):
      plt.text(c, h, str(h * width + c), color='k', size=max_text_size, **text_params)

  arrow_h_offset = 0.25  # data coordinates, empirically determined
  arrow_length = 1 - 2 * arrow_h_offset
  head_width = 2.5 * max_arrow_width
  head_length = 2 * max_arrow_width
  arrow_params = {'length_includes_head': True, 'shape': 'right',
                  'head_starts_at_zero': True}

  def draw_arrow(d1, d2, weight=0.3):
    # color
    fc = get_color_by_weight(weight)
    ec = fc

    # position
    a = 0.1
    where = get_position(d1, d2)
    if where == 'left':
      x_pos, y_pos = d1[1] - arrow_sep, d1[0] + arrow_h_offset
      x, y, rotation = d1[1] - arrow_sep - a, d1[0] + arrow_h_offset / 2 * 4, 90
    elif where == 'right':
      x_pos, y_pos = d1[1] + arrow_sep, d1[0] - arrow_h_offset
      x, y, rotation = d1[1] + arrow_sep + a, d1[0] - arrow_h_offset / 2 * 4, 90
    elif where == 'top':
      x_pos, y_pos = d1[1] + arrow_h_offset, d1[0] + arrow_sep
      x, y, rotation = d1[1] + arrow_h_offset / 2 * 4, d1[0] + arrow_sep + a, 0
    elif where == 'bottom':
      x_pos, y_pos = d1[1] - arrow_h_offset, d1[0] - arrow_sep
      x, y, rotation = d1[1] - arrow_h_offset / 2 * 4, d1[0] - arrow_sep - a, 0
    else:
      raise ValueError

    # arrow
    x_scale, y_scale = (d2[1] - d1[1]), (d2[0] - d1[0])
    plt.arrow(x_pos, y_pos, x_scale * arrow_length, y_scale * arrow_length,
              fc=fc, ec=ec, alpha=alpha, width=max_arrow_width,
              head_width=head_width, head_length=head_length,
              **arrow_params)

    # text
    plt.text(x, y, "{:.4f}".format(weight), size=label_text_size, ha='center',
             va='center', color=fc, rotation=rotation)

  for ii, jj, ww in zip(i, j, w):
    a = (ii // width, ii % width)
    b = (jj // width, jj % width)
    draw_arrow(a, b, ww)

  if show:
    plt.show()


#####
import brainpy.math as bm


def gaussian1(size, ranges, seed=None):
  rng = bm.random.RandomState(seed)
  return rng.truncated_normal(ranges[0], ranges[1], size)


def gaussian2(mu, sigma, size, ranges, seed=None):
  rng = np.random.RandomState(seed=seed)
  min_, max_ = ranges[0], ranges[1]
  all_gT = []
  for i in range(size):
    r = rng.normal(mu, sigma)
    while not min_ <= r <= max_:
      r = rng.normal(mu, sigma)
    all_gT.append(r)
  return bm.array(all_gT)


def gaussian3(mu, sigma, size, ranges, rng: bm.random.RandomState):
  return rng.truncated_normal((ranges[0] - mu) / sigma,
                              (ranges[1] - mu) / sigma,
                              size) * sigma + mu


def uniform1(size, ranges, seed=None):
  rng = np.random.RandomState(seed=seed)
  min_, max_ = ranges[0], ranges[1]
  all_gT = np.linspace(min_, max_, num=bp.tools.size2num(size))
  for _ in range(20):
    rng.shuffle(all_gT)
  return bm.asarray(all_gT)


def uniform2(size, ranges, rng):
  min_, max_ = ranges[0], ranges[1]
  all_gT = bm.linspace(min_, max_, num=bp.tools.size2num(size))
  for _ in range(10):
    rng.shuffle(all_gT)
  return all_gT


def cycleTimeCal(data, interval_thre):
  # 获取极小值
  from scipy.signal import argrelextrema
  min_peaks = argrelextrema(data, np.less)
  min_peaks = np.array(min_peaks)
  # interval_thre = 1/4

  min_peaks_new = []
  for min_peak in min_peaks[0, :]:
    # print(max_peak)
    if data[min_peak] < (data.mean() + (data.max() - data.mean()) * 0):
      # print(data[min_peak])
      min_peaks_new.append(min_peak)
    peak_interval = []
    for i, min_peak in enumerate(min_peaks_new):
      if i > 0:
        peak_interval.append(min_peak - min_peaks_new[i - 1])
    min_peaks_new1 = []
    for i, min_peak in enumerate(min_peaks_new):
      if i == 0:
        min_peaks_new1.append(min_peak)
      elif min_peak - min_peaks_new[i - 1] > interval_thre * np.mean(peak_interval):
        min_peaks_new1.append(min_peak)
      elif data[min_peak] < data[min_peaks_new[i - 1]]:
        del (min_peaks_new1[-1])
        min_peaks_new1.append(min_peak)

  cycleTime = (min_peaks_new1[-1] - min_peaks_new1[0]) / (len(min_peaks_new1) - 1) * bm.get_dt()
  x0 = min_peaks_new1[:-1]
  x1 = min_peaks_new1[1:]
  interval_peak = np.array(x1) - np.array(x0)
  std_cycleTime = np.std(interval_peak)

  # 绘制极值点图像
  # plt.figure(figsize = (18,6))
  # plt.plot(data)
  # plt.scatter(min_peaks_new1, data[min_peaks_new1], c='b', label='Min Peaks')
  # plt.legend()
  # plt.xlabel('time (s)')
  # plt.ylabel('Amplitude')
  # plt.title("Find Peaks")

  return cycleTime, std_cycleTime


def signalGenerate(time, cycle):
  t = np.linspace(0, time, time) * bm.get_dt()
  # w = pi/2
  fs = 2 * math.pi / cycle
  x1 = np.sin(fs * t)
  return x1


def size2len(size):
  if isinstance(size, int):
    return size
  elif isinstance(size, (tuple, list)):
    a = 1
    for b in size:
      a *= b
    return a
  else:
    raise ValueError


def vectorize_map(func: callable,
                  arguments: Union[Dict, Sequence],
                  num_parallel: int,
                  clear_buffer: bool = False):
  if not isinstance(arguments, (dict, tuple, list)):
    raise TypeError(f'"arguments" must be sequence or dict, but we got {type(arguments)}')
  elements, tree = tree_flatten(arguments, is_leaf=lambda a: isinstance(a, bm.JaxArray))
  if clear_buffer:
    elements = [np.asarray(ele) for ele in elements]
  num_pars = [len(ele) for ele in elements]
  if len(np.unique(num_pars)) != 1:
    raise ValueError(f'All elements in parameters should have the same length. '
                     f'But we got {tree_unflatten(tree, num_pars)}')

  res_tree = None
  results = None
  vmap_func = vmap(func)
  for i in range(0, num_pars[0], num_parallel):
    run_f = vmap(func) if clear_buffer else vmap_func
    if isinstance(arguments, dict):
      r = run_f(**tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    else:
      r = run_f(*tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    res_values, res_tree = tree_flatten(r, is_leaf=lambda a: isinstance(a, bm.JaxArray))
    if results is None:
      results = tuple([np.asarray(val) if clear_buffer else val] for val in res_values)
    else:
      for j, val in enumerate(res_values):
        results[j].append(np.asarray(val) if clear_buffer else val)
    if clear_buffer:
      bm.clear_buffer_memory()
  if res_tree is None:
    return None
  results = ([np.concatenate(res, axis=0) for res in results]
             if clear_buffer else
             [bm.concatenate(res, axis=0) for res in results])
  return tree_unflatten(res_tree, results)


def parallelize_map(func: callable,
                    arguments: Union[Dict, Sequence],
                    num_parallel: int,
                    clear_buffer: bool = False):
  if not isinstance(arguments, (dict, tuple, list)):
    raise TypeError(f'"arguments" must be sequence or dict, but we got {type(arguments)}')
  elements, tree = tree_flatten(arguments, is_leaf=lambda a: isinstance(a, bm.JaxArray))
  if clear_buffer:
    elements = [np.asarray(ele) for ele in elements]
  num_pars = [len(ele) for ele in elements]
  if len(np.unique(num_pars)) != 1:
    raise ValueError(f'All elements in parameters should have the same length. '
                     f'But we got {tree_unflatten(tree, num_pars)}')

  res_tree = None
  results = None
  vmap_func = pmap(func)
  for i in range(0, num_pars[0], num_parallel):
    run_f = pmap(func) if clear_buffer else vmap_func
    if isinstance(arguments, dict):
      r = run_f(**tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    else:
      r = run_f(*tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    res_values, res_tree = tree_flatten(r, is_leaf=lambda a: isinstance(a, bm.JaxArray))
    if results is None:
      results = tuple([np.asarray(val) if clear_buffer else val] for val in res_values)
    else:
      for j, val in enumerate(res_values):
        results[j].append(np.asarray(val) if clear_buffer else val)
    if clear_buffer:
      bm.clear_buffer_memory()
  if res_tree is None:
    return None
  results = ([np.concatenate(res, axis=0) for res in results]
             if clear_buffer else
             [bm.concatenate(res, axis=0) for res in results])
  return tree_unflatten(res_tree, results)


def detect_unique_name(name: str, postfix: str):
  assert postfix[0] == '.'
  i = 0
  version = ''
  while os.path.exists(name + version + postfix):
    i += 1
    version = f'-v{i}'

  return name + version + postfix
