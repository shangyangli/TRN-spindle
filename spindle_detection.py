# -*- coding: utf-8 -*-
# %% [markdown]
"""
# Automatic Spindle Detection
"""

# %%
import logging

import mne
from mne.filter import filter_data
import numpy as np
import pandas as pd

from scipy import signal
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from yasa.io import set_log_level
from yasa.numba import _detrend, _rms
from yasa.others import (moving_transform, trimbothstd, _merge_close)
from yasa.spectral import stft_power

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# %%
import sys

sys.path.append('../')

import utils

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
logger = logging.getLogger('yasa')

__all__ = ['spindles_detect', 'SpindlesResults']


# %% [markdown]
# ## Data preprocessing

# %%
def _check_data_hypno(data, sf=None, ch_names=None, hypno=None, include=None,
                      check_amp=True):
    """Helper functions for preprocessing of data and hypnogram."""
    # 1) Extract data as a 2D NumPy array
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        data = data.get_data() * 1e6  # Convert from V to uV
    else:
        assert sf is not None, 'sf must be specified if not using MNE Raw.'
    data = np.asarray(data, dtype=np.float64)
    assert data.ndim in [1, 2], 'data must be 1D (times) or 2D (chan, times).'
    if data.ndim == 1:
        # Force to 2D array: (n_chan, n_samples)
        data = data[None, ...]
    n_chan, n_samples = data.shape

    # 2) Check channel names
    if ch_names is None:
        ch_names = ['CHAN' + str(i).zfill(3) for i in range(n_chan)]
    else:
        assert len(ch_names) == n_chan

    # 3) Check hypnogram
    if hypno is not None:
        hypno = np.asarray(hypno, dtype=int)
        assert hypno.ndim == 1, 'Hypno must be one dimensional.'
        assert hypno.size == n_samples, 'Hypno must have same size as data.'
        unique_hypno = np.unique(hypno)
        logger.info('Number of unique values in hypno = %i', unique_hypno.size)
        assert include is not None, 'include cannot be None if hypno is given'
        include = np.atleast_1d(np.asarray(include))
        assert include.size >= 1, '`include` must have at least one element.'
        assert hypno.dtype.kind == include.dtype.kind, ('hypno and include '
                                                        'must have same dtype')
        assert np.in1d(hypno, include).any(), ('None of the stages specified '
                                               'in `include` are present in '
                                               'hypno.')

    # 4) Check data amplitude
    logger.info('Number of samples in data = %i', n_samples)
    logger.info('Sampling frequency = %.2f Hz', sf)
    logger.info('Data duration = %.2f seconds', n_samples / sf)
    all_ptp = np.ptp(data, axis=-1)
    all_trimstd = trimbothstd(data, cut=0.05)
    bad_chan = np.zeros(n_chan, dtype=bool)
    for i in range(n_chan):
        logger.info('Trimmed standard deviation of %s = %.4f uV'
                    % (ch_names[i], all_trimstd[i]))
        logger.info('Peak-to-peak amplitude of %s = %.4f uV'
                    % (ch_names[i], all_ptp[i]))
        if check_amp and not (0.1 < all_trimstd[i] < 1e3):
            logger.error('Wrong data amplitude for %s '
                         '(trimmed STD = %.3f). Unit of data MUST be uV! '
                         'Channel will be skipped.'
                         % (ch_names[i], all_trimstd[i]))
            bad_chan[i] = True

    # 5) Create sleep stage vector mask
    if hypno is not None:
        mask = np.in1d(hypno, include)
    else:
        mask = np.ones(n_samples, dtype=bool)

    return (data, sf, ch_names, hypno, include, mask, n_chan, n_samples,
            bad_chan)


# %% [markdown]
# ## Basic Detection Methods

# %%
class _DetectionResults(object):
    """Main class for detection results."""

    def __init__(self, events, data, sf, ch_names, hypno, data_filt):
        self._events = events
        self._data = data
        self._sf = sf
        self._hypno = hypno
        self._ch_names = ch_names
        self._data_filt = data_filt

    def get_mask(self):
        """get_mask"""
        from yasa.others import _index_to_events
        mask = np.zeros(self._data.shape, dtype=int)
        for i in self._events['IdxChannel'].unique():
            ev_chan = self._events[self._events['IdxChannel'] == i]
            idx_ev = _index_to_events(
                ev_chan[['Start', 'End']].to_numpy() * self._sf)
            mask[i, idx_ev] = 1
        return np.squeeze(mask)

    def summary(self, event_type, grp_chan=False, grp_stage=False, aggfunc='mean', sort=True):
        """Summary"""
        grouper = []
        if grp_stage is True and 'Stage' in self._events:
            grouper.append('Stage')
        if grp_chan is True and 'Channel' in self._events:
            grouper.append('Channel')
        if not len(grouper):
            return self._events.copy()

        if event_type == 'spindles':
            aggdict = {'Start': 'count',
                       'Duration': aggfunc,
                       'Amplitude': aggfunc,
                       'RMS': aggfunc,
                       'AbsPower': aggfunc,
                       'RelPower': aggfunc,
                       'Frequency': aggfunc,
                       'Oscillations': aggfunc,
                       'Symmetry': aggfunc}

        elif event_type == 'sw':
            aggdict = {'Start': 'count',
                       'Duration': aggfunc,
                       'ValNegPeak': aggfunc,
                       'ValPosPeak': aggfunc,
                       'PTP': aggfunc,
                       'Slope': aggfunc,
                       'Frequency': aggfunc}

            if 'PhaseAtSigmaPeak' in self._events:
                from scipy.stats import circmean
                aggdict['PhaseAtSigmaPeak'] = lambda x: circmean(x, low=-np.pi,
                                                                 high=np.pi)
                aggdict['ndPAC'] = aggfunc

        else:  # REM
            aggdict = {'Start': 'count',
                       'Duration': aggfunc,
                       'LOCAbsValPeak': aggfunc,
                       'ROCAbsValPeak': aggfunc,
                       'LOCAbsRiseSlope': aggfunc,
                       'ROCAbsRiseSlope': aggfunc,
                       'LOCAbsFallSlope': aggfunc,
                       'ROCAbsFallSlope': aggfunc}

        # Apply grouping
        df_grp = self._events.groupby(grouper, sort=sort,
                                      as_index=False).agg(aggdict)
        df_grp = df_grp.rename(columns={'Start': 'Count'})

        # Calculate density (= number per min of each stage)
        if self._hypno is not None and grp_stage is True:
            stages = np.unique(self._events['Stage'])
            dur = {}
            for st in stages:
                # Get duration in minutes of each stage present in dataframe
                dur[st] = self._hypno[self._hypno == st].size / (60 * self._sf)

            # Insert new density column in grouped dataframe after count
            df_grp.insert(
                loc=df_grp.columns.get_loc('Count') + 1, column='Density',
                value=df_grp.apply(lambda rw: rw['Count'] / dur[rw['Stage']],
                                   axis=1))

        return df_grp.set_index(grouper)

    def get_sync_events(self, center, time_before, time_after, filt=(None, None)):
        """Get_sync_events
        (not for REM, spindles & SW only)
        """
        from yasa.others import get_centered_indices
        assert time_before >= 0
        assert time_after >= 0
        bef = int(self._sf * time_before)
        aft = int(self._sf * time_after)
        # TODO: Step size is determined by sf: 0.01 sec at 100 Hz, 0.002 sec at
        # 500 Hz, 0.00390625 sec at 256 Hz. Should we add a step_size=0.01
        # option?
        time = np.arange(-bef, aft + 1, dtype='int') / self._sf

        if any(filt):
            data = mne.filter.filter_data(self._data, self._sf, l_freq=filt[0],
                                          h_freq=filt[1], method='fir',
                                          verbose=False)
        else:
            data = self._data

        df_sync = pd.DataFrame()

        for i in self._events['IdxChannel'].unique():
            ev_chan = self._events[self._events['IdxChannel'] == i].copy()
            ev_chan['Event'] = np.arange(ev_chan.shape[0])
            peaks = (ev_chan[center] * self._sf).astype(int).to_numpy()
            # Get centered indices
            idx, idx_valid = get_centered_indices(data[i, :], peaks, bef, aft)
            # If no good epochs are returned raise a warning
            if len(idx_valid) == 0:
                logger.error(
                    'Time before and/or time after exceed data bounds, please '
                    'lower the temporal window around center. '
                    'Skipping channel.'
                )
                continue

            # Get data at indices and time vector and convert to df
            amps = data[i, idx]
            df_chan = pd.DataFrame(amps.T)
            df_chan['Time'] = time
            # Convert to long-format
            df_chan = df_chan.melt(id_vars='Time', var_name='Event',
                                   value_name='Amplitude')
            # Append stage
            if 'Stage' in self._events:
                df_chan = df_chan.merge(
                    ev_chan[['Event', 'Stage']].iloc[idx_valid]
                )
            # Append channel name
            df_chan['Channel'] = ev_chan['Channel'].iloc[0]
            df_chan['IdxChannel'] = i
            # Append to master dataframe
            df_sync = df_sync.append(df_chan, ignore_index=True)

        return df_sync

    def plot_average(self, event_type, center='Peak', hue='Channel', time_before=1, time_after=1, filt=(None, None),
                     figsize=(6, 4.5), **kwargs):
        """plot_average
        (not for REM, spindles & SW only)
        """
        df_sync = self.get_sync_events(center=center, time_before=time_before,
                                       time_after=time_after, filt=filt)
        assert not df_sync.empty, "Could not calculate event-locked data."
        assert hue in ['Stage', 'Channel'], "hue must be 'Channel' or 'Stage'"
        assert hue in df_sync.columns, "%s is not present in data." % hue

        if event_type == 'spindles':
            title = "Average spindle"
        else:  # "sw":
            title = "Average SW"

        # Start figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df_sync, x='Time', y='Amplitude', hue=hue, ax=ax,
                     **kwargs)
        # ax.legend(frameon=False, loc='lower right')
        ax.set_xlim(df_sync['Time'].min(), df_sync['Time'].max())
        ax.set_title(title)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude (uV)')
        plt.show()
        return ax

    def plot_detection(self):
        """Plot an overlay of the detected events on the signal."""
        import ipywidgets as ipy

        # Define mask
        sf = self._sf
        win_size = 10
        mask = self.get_mask()
        highlight = self._data * mask
        highlight = np.where(highlight == 0, np.nan, highlight)
        highlight_filt = self._data_filt * mask
        highlight_filt = np.where(highlight_filt == 0, np.nan, highlight_filt)

        n_epochs = int((self._data.shape[-1] / sf) / win_size)
        times = np.arange(self._data.shape[-1]) / sf

        # Define xlim and xrange
        xlim = [0, win_size]
        xrng = np.arange(xlim[0] * sf, (xlim[1] * sf + 1), dtype=int)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        plt.plot(times[xrng], self._data[0, xrng], 'k', lw=1)
        plt.plot(times[xrng], highlight[0, xrng], 'indianred')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (uV)')
        fig.canvas.header_visible = False
        fig.tight_layout()

        # WIDGETS
        layout = ipy.Layout(
            width="50%",
            justify_content='center',
            align_items='center'
        )

        sl_ep = ipy.IntSlider(
            min=0,
            max=n_epochs,
            step=1,
            value=0,
            layout=layout,
            description="Epoch:",
        )

        sl_amp = ipy.IntSlider(
            min=25,
            max=500,
            step=25,
            value=150,
            layout=layout,
            orientation='horizontal',
            description="Amplitude:"
        )

        dd_ch = ipy.Dropdown(
            options=self._ch_names, value=self._ch_names[0],
            description='Channel:'
        )

        dd_win = ipy.Dropdown(
            options=[1, 5, 10, 30, 60],
            value=win_size,
            description='Window size:',
        )

        dd_check = ipy.Checkbox(
            value=False,
            description='Filtered',
        )

        def update(epoch, amplitude, channel, win_size, filt):
            """Update plot."""
            n_epochs = int((self._data.shape[-1] / sf) / win_size)
            sl_ep.max = n_epochs
            xlim = [epoch * win_size, (epoch + 1) * win_size]
            xrng = np.arange(xlim[0] * sf, (xlim[1] * sf), dtype=int)
            # Check if filtered
            data = self._data if not filt else self._data_filt
            overlay = highlight if not filt else highlight_filt
            try:
                ax.lines[0].set_data(times[xrng], data[dd_ch.index, xrng])
                ax.lines[1].set_data(times[xrng], overlay[dd_ch.index, xrng])
                ax.set_xlim(xlim)
            except IndexError:
                pass
            ax.set_ylim([-amplitude, amplitude])

        return ipy.interact(update, epoch=sl_ep, amplitude=sl_amp,
                            channel=dd_ch, win_size=dd_win, filt=dd_check)

# %%
def gau_func(x, mu, sig, amp):
    return amp * np.exp(-((x - mu) / sig) ** 2 / 2) / (sig * np.sqrt(2 * np.pi))


# %%
def spindles_detect(data,
                    sf=None,
                    ch_names=None,
                    hypno=None,
                    include=(1, 2, 3),
                    freq_sp=(12, 15),
                    freq_broad=(1, 30),
                    min_distance=250,
                    thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                    corr_window=0.1,
                    corr_step=0.05,
                    rms_cut=0.025,
                    soft_width=0.1,
                    sp_threshold=2.01,
                    gaussian_validate=True, 
                    multi_only=False,
                    verbose=False,
                    show=False):
    """Spindles detection.

    Parameters
    ----------
    data : array_like
        Single or multi-channel data. Unit must be uV and shape (n_samples) or
        (n_chan, n_samples). Can also be a :py:class:`mne.io.BaseRaw`,
        in which case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be automatically converted from
        Volts (MNE) to micro-Volts (YASA).
    sf : float
        Sampling frequency of the data in Hz.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.

        .. tip:: If the detection is taking too long, make sure to downsample
            your data to 100 Hz (or 128 Hz). For more details, please refer to
            :py:func:`mne.filter.resample`.
    ch_names : list of str
        Channel names. Can be omitted if ``data`` is a
        :py:class:`mne.io.BaseRaw`.
    hypno : array_like
        Sleep stage (hypnogram). If the hypnogram is loaded, the
        detection will only be applied to the value defined in
        ``include`` (default = N1 + N2 + N3 sleep).

        The hypnogram must have the same number of samples as ``data``.
        To upsample your hypnogram, please refer to
        :py:func:`yasa.hypno_upsample_to_data`.

        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    include : tuple, list or int
        Values in ``hypno`` that will be included in the mask. The default is
        (1, 2, 3), meaning that the detection is applied on N1, N2 and N3
        sleep. This has no effect when ``hypno`` is None.
    freq_sp : tuple or list
        Spindles frequency range. Default is 12 to 15 Hz. Please note that YASA
        uses a FIR filter (implemented in MNE) with a 1.5Hz transition band,
        which means that for `freq_sp = (12, 15 Hz)`, the -6 dB points are
        located at 11.25 and 15.75 Hz.
    freq_broad : tuple or list
        Broad band frequency range. Default is 1 to 30 Hz.
    min_distance : int
        If two spindles are closer than ``min_distance`` (in ms), they are
        merged into a single spindles. Default is 500 ms.
    thresh : dict
        Detection thresholds:

        * ``'rel_pow'``: Relative power (= power ratio freq_sp / freq_broad).
        * ``'corr'``: Moving correlation between original signal and
          sigma-filtered signal.
        * ``'rms'``: Number of standard deviations above the mean of a moving
          root mean square of sigma-filtered signal.

        You can disable one or more threshold by putting ``None`` instead:

        .. code-block:: python

            thresh = {'rel_pow': None, 'corr': 0.65, 'rms': 1.5}
            thresh = {'rel_pow': None, 'corr': None, 'rms': 3}
    multi_only : boolean
        Define the behavior of the multi-channel detection. If True, only
        spindles that are present on at least two channels are kept. If False,
        no selection is applied and the output is just a concatenation of the
        single-channel detection dataframe. Default is False.

    Notes
    -----
    The parameters that are calculated for each spindle are:

    * ``'Start'``: Start time of the spindle, in seconds from the beginning of
      data.
    * ``'Peak'``: Time at the most prominent spindle peak (in seconds).
    * ``'End'`` : End time (in seconds).
    * ``'Duration'``: Duration (in seconds)
    * ``'Amplitude'``: Peak-to-peak amplitude of the (detrended) spindle in
      the raw data (in µV).
    * ``'RMS'``: Root-mean-square (in µV)
    * ``'AbsPower'``: Median absolute power (in log10 µV^2),
      calculated from the Hilbert-transform of the ``freq_sp`` filtered signal.
    * ``'RelPower'``: Median relative power of the ``freq_sp`` band in spindle
      calculated from a short-term fourier transform and expressed as a
      proportion of the total power in ``freq_broad``.
    * ``'Frequency'``: Median instantaneous frequency of spindle (in Hz),
      derived from an Hilbert transform of the ``freq_sp`` filtered signal.
    * ``'Oscillations'``: Number of oscillations (= number of positive peaks
      in spindle.)
    * ``'Symmetry'``: Location of the most prominent peak of spindle,
      normalized from 0 (start) to 1 (end). Ideally this value should be close
      to 0.5, indicating that the most prominent peak is halfway through the
      spindle.
    * ``'Stage'`` : Sleep stage during which spindle occured, if ``hypno``
      was provided.

      All parameters are calculated from the broadband-filtered EEG
      (frequency range defined in ``freq_broad``).

    For better results, apply this detection only on artefact-free NREM sleep.

    References
    ----------
    The sleep spindles detection algorithm is based on:

    * Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., & Warby, S. C.
      (2018). `A sleep spindle detection algorithm that emulates human expert
      spindle scoring. <https://doi.org/10.1016/j.jneumeth.2018.08.014>`_
      Journal of Neuroscience Methods.

    """
    set_log_level(verbose)

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan
     ) = _check_data_hypno(data, sf, ch_names, hypno, include)

    # If all channels are bad
    if sum(bad_chan) == n_chan:
        logger.warning('All channels have bad amplitude. Returning None.')
        return None

    # Check detection thresholds
    if 'rel_pow' not in thresh.keys():
        thresh['rel_pow'] = 0.20
    if 'corr' not in thresh.keys():
        thresh['corr'] = 0.65
    if 'rms' not in thresh.keys():
        thresh['rms'] = 1.5

    times = np.arange(data.shape[1]) * sf

    # Filtering
    nfast = next_fast_len(n_samples)
    # 1) Broadband bandpass filter (optional -- careful of lower freq for PAC)
    data_broad = filter_data(data, sf, freq_broad[0], freq_broad[1],
                             method='fir', verbose=0)
    # 2) Sigma bandpass filter
    # The width of the transition band is set to 1.5 Hz on each side,
    # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located at
    # 11.25 and 15.75 Hz.
    data_sigma = filter_data(data, sf, freq_sp[0], freq_sp[1],
                             l_trans_bandwidth=1.5, h_trans_bandwidth=1.5,
                             method='fir', verbose=0)



    # Hilbert power (to define the instantaneous frequency / power)
    analytic = signal.hilbert(data_sigma, N=nfast)[:, :n_samples]
    inst_phase = np.angle(analytic)
    inst_pow = np.square(np.abs(analytic))
    inst_freq = (sf / (2 * np.pi) * np.diff(inst_phase, axis=-1))

    # Initialize empty output dataframe
    df = pd.DataFrame()

    for i in range(n_chan):
        if show:
            plt.figure(figsize=(15, 5))
            plt.plot(times, data_broad[i], label='broad')
            plt.plot(times, data_sigma[i], label='sigma')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude (uV)')
            _ = plt.xlim(0, times[-1])
            plt.legend()
            plt.tight_layout()
            plt.title('Power')

        # ####################################################################
        # START SINGLE CHANNEL DETECTION
        # ####################################################################

        # First, skip channels with bad data amplitude
        if bad_chan[i]:
            continue

        # Compute the pointwise relative power using interpolated STFT
        # Here we use a step of 200 ms to speed up the computation.
        # Note that even if the threshold is None we still need to calculate it
        # for the individual spindles parameter (RelPow).
        f, t, Sxx = stft_power(data_broad[i, :], sf, window=2, step=.2,
                               band=freq_broad, interp=False, norm=True)
        idx_sigma = np.logical_and(f >= freq_sp[0], f <= freq_sp[1])
        rel_pow = Sxx[idx_sigma].sum(0)

        if show:
            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
            plt.subplots_adjust(hspace=.25)
            im = ax1.pcolormesh(t, f, Sxx, cmap='Spectral_r', vmax=0.2)
            ax1.set_title('Spectrogram')
            ax1.set_ylabel('Frequency (Hz)')
            ax2.plot(t, rel_pow)
            ax2.set_ylabel('Relative power (% $uV^2$)')
            ax2.set_xlim(t[0], t[-1])
            ax2.set_xlabel('Time (sec)')
            ax2.axhline(thresh['rel_pow'], ls=':', lw=2, color='indianred', label='Threshold #1')
            plt.legend()
            _ = ax2.set_title('Relative power in the sigma band')
            plt.tight_layout()

        # Let's interpolate `rel_pow` to get one value per sample
        # Note that we could also have use the `interp=True` in the
        # `stft_power` function, however 2D interpolation is much slower than
        # 1D interpolation.
        func = interp1d(t, rel_pow, kind='cubic', bounds_error=False, fill_value=0)
        t = np.arange(n_samples) / sf
        rel_pow = func(t)

        # POINT: Moving correlation
        _, mcorr = moving_transform(x=data_sigma[i, :], y=data_broad[i, :],
                                    sf=sf, window=corr_window, step=corr_step,
                                    method='corr', interp=True)

        if show:
            plt.figure(figsize=(15, 5))
            plt.plot(times, mcorr)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Pearson correlation')
            plt.axhline(thresh['corr'], ls=':', lw=2, color='indianred', label='Threshold #2')
            plt.legend()
            plt.title('Moving correlation between $EEG_{bf}$ and $EEG_{\sigma}$')
            _ = plt.xlim(0, times[-1])
            plt.tight_layout()

        # POINT: Moving rms
        _, mrms = moving_transform(x=data_sigma[i, :], sf=sf,
                                   window=.3, step=.1, method='rms',
                                   interp=True)

        # Let's define the thresholds
        if hypno is None:
            thresh_rms = mrms.mean() + thresh['rms'] * trimbothstd(mrms, cut=rms_cut)
        else:
            thresh_rms = mrms[mask].mean() + thresh['rms'] * trimbothstd(mrms[mask], cut=rms_cut)
        # Avoid too high threshold caused by Artefacts / Motion during Wake
        thresh_rms = min(thresh_rms, 10)
        logger.info('Moving RMS threshold = %.3f', thresh_rms)

        if show:
            plt.figure(figsize=(15, 5))
            plt.plot(times, mrms)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Root mean square')
            plt.axhline(thresh_rms, ls=':', lw=2, color='indianred', label='Threshold #3')
            plt.legend()
            plt.title('Moving RMS of $EEG_{\sigma}$')
            _ = plt.xlim(0, times[-1])
            plt.tight_layout()

        # POINT: Boolean vector of supra-threshold indices
        # relative power
        idx_sum = np.zeros(n_samples)
        idx_rel_pow = (rel_pow >= thresh['rel_pow']).astype(int)
        idx_sum += idx_rel_pow
        logger.info('N supra-theshold relative power = %i', idx_rel_pow.sum())
        # moving correlation
        idx_mcorr = (mcorr >= thresh['corr']).astype(int)
        idx_sum += idx_mcorr
        logger.info('N supra-theshold moving corr = %i', idx_mcorr.sum())
        # moving RMS
        idx_mrms = (mrms >= thresh_rms).astype(int)
        idx_sum += idx_mrms
        logger.info('N supra-theshold moving RMS = %i', idx_mrms.sum())

        # Make sure that we do not detect spindles outside mask
        if hypno is not None:
            idx_sum[~mask] = 0

        # POINT: soft threshold

        # The detection using the three thresholds tends to underestimate the
        # real duration of the spindle. To overcome this, we compute a soft
        # threshold by smoothing the idx_sum vector with a 100 ms window.
        w = int(soft_width * sf)
        idx_sum = np.convolve(idx_sum, np.ones(w) / w, mode='same')

        if show:
            plt.figure(figsize=(15, 5))
            plt.plot(times, idx_sum, '.-', markersize=5)
            plt.fill_between(times, sp_threshold, idx_sum,
                             where=idx_sum > sp_threshold,
                             color='indianred', alpha=.8)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Number of passed thresholds')
            plt.title('Decision function')
            _ = plt.xlim(0, times[-1])
            plt.tight_layout()

        # And we then find indices that are strictly greater than 2, i.e. we
        # find the 'true' beginning and 'true' end of the events by finding
        # where at least two out of the three treshold were crossed.
        where_sp = np.where(idx_sum > sp_threshold)[0]

        # If no events are found, skip to next channel
        if not len(where_sp):
            logger.warning('No spindle were found in channel %s.', ch_names[i])
            continue

        # Merge events that are too close
        if min_distance is not None and min_distance > 0:
            where_sp = _merge_close(where_sp, min_distance, sf)

        # POINT: Fit the data with the Gaussian function,
        # and extract start, end, and duration of each spindle
        split_spendles = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
        sp_start, sp_end, spindles = [], [], []
        if len(split_spendles) and show:
            plt.figure(figsize=(15, 5))
            plt.plot(data_sigma[i], 'k', label='origin')
            plt.ylabel('Amplitude (uV)')
            _ = plt.xlim(0, times.size)
        max_ylim = data_sigma[i].max() * 1.1
        min_ylim = data_sigma[i].min() * 1.1
        for sp_i, sp in enumerate(split_spendles):
            if len(sp) >= 2:
                start, end = sp[0], sp[-1]
                sp_data = data_sigma[i, start: end]
                maxima_idx = utils.find_local_maxima(sp_data)
                minima_idx = utils.find_local_minima(sp_data)
                if len(maxima_idx) >= len(minima_idx):
                    sp_idx = maxima_idx
                    sp_to_fit = sp_data[sp_idx]
                else:
                    sp_idx = minima_idx
                    sp_to_fit = -sp_data[sp_idx]
                sp_xs = np.arange(len(sp_to_fit))

                if show:
                    index = np.arange(start, end)
                    plt.plot(index, sp_data, 'g')
                    if sp_i % 2 == 0:
                        y_pos = sp_data.max() * 1.45
                        if max_ylim < y_pos: max_ylim = y_pos
                    else:
                        y_pos = sp_data.min() * 1.45
                        if min_ylim > y_pos: min_ylim = y_pos
                    plt.text((start + end) / 2, y_pos, f'Spindle {sp_i + 1}',
                             ha='center', va='center', fontsize=15, color='red')

                if gaussian_validate:
                    if len(sp_xs) > 3:
                        try:
                            popt, pcov = curve_fit(gau_func, sp_xs, sp_to_fit)
                            if np.all(np.diag(pcov) <= 100):
                                sp_start.append(start / sf)
                                sp_end.append(end / sf)
                                spindles.append(sp)
                                if show:
                                    sp_ys = gau_func(sp_xs, *popt)
                                    plt.plot(index[sp_idx], sp_ys, 'r')
                                    text = f'cov: {pcov[0, 0]:.2f}, {pcov[1, 1]:.2f}, {pcov[2, 2]:.2f}'
                                    plt.text((start + end) / 2,
                                             (sp_data.max() if i % 2 == 0 else sp_data.min()) * 1.1,
                                             text, ha='center', va='center', fontsize=10, color='C1')
                        except RuntimeError:
                            pass
                else:
                    sp_start.append(start / sf)
                    sp_end.append(end / sf)
                    spindles.append(sp)
        if show:
            if len(split_spendles):
                plt.ylim(min_ylim, max_ylim)
            plt.tight_layout()
            plt.show()

        sp_start = np.array(sp_start)
        sp_end = np.array(sp_end)

        # Initialize empty variables
        sp_amp = np.zeros(len(spindles))
        sp_freq = np.zeros(len(spindles))
        sp_rms = np.zeros(len(spindles))
        sp_osc = np.zeros(len(spindles))
        sp_sym = np.zeros(len(spindles))
        sp_abs = np.zeros(len(spindles))
        sp_rel = np.zeros(len(spindles))
        sp_sta = np.zeros(len(spindles))
        sp_pro = np.zeros(len(spindles))
        # sp_cou = np.zeros(len(sp))

        # Number of oscillations (number of peaks separated by at least 60 ms)
        # --> 60 ms because 1000 ms / 16 Hz = 62.5 m, in other words, at 16 Hz,
        # peaks are separated by 62.5 ms. At 11 Hz peaks are separated by 90 ms
        distance = 60 * sf / 1000

        for j in np.arange(len(spindles)):
            # Important: detrend the signal to avoid wrong PTP amplitude
            sp_x = np.arange(data_broad[i, spindles[j]].size, dtype=np.float64)
            sp_det = _detrend(sp_x, data_broad[i, spindles[j]])
            # sp_det = signal.detrend(data_broad[i, sp[i]], type='linear')
            sp_amp[j] = np.ptp(sp_det)  # Peak-to-peak amplitude
            sp_rms[j] = _rms(sp_det)  # Root mean square
            sp_rel[j] = np.median(rel_pow[spindles[j]])  # Median relative power

            # Hilbert-based instantaneous properties
            sp_inst_freq = inst_freq[i, spindles[j]]
            sp_inst_pow = inst_pow[i, spindles[j]]
            sp_abs[j] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
            sp_freq[j] = np.median(sp_inst_freq[sp_inst_freq > 0])

            # Number of oscillations
            peaks, peaks_params = signal.find_peaks(sp_det,
                                                    distance=distance,
                                                    prominence=(None, None))
            sp_osc[j] = len(peaks)

            # For frequency and amplitude, we can also optionally use these
            # faster alternatives. If we use them, we do not need to compute
            # the Hilbert transform of the filtered signal.
            # sp_freq[j] = sf / np.mean(np.diff(peaks))
            # sp_amp[j] = peaks_params['prominences'].max()

            # Peak location & symmetry index
            # pk is expressed in sample since the beginning of the spindle
            pk = peaks[peaks_params['prominences'].argmax()]
            sp_pro[j] = sp_start[j] + pk / sf
            sp_sym[j] = pk / sp_det.size

            # Sleep stage
            if hypno is not None:
                sp_sta[j] = hypno[spindles[j]][0]

        # Create a dataframe
        sp_params = {'Start': sp_start,
                     'Peak': sp_pro,
                     'End': sp_end,
                     'Duration': sp_end - sp_start,
                     'Amplitude': sp_amp,
                     'RMS': sp_rms,
                     'AbsPower': sp_abs,
                     'RelPower': sp_rel,
                     'Frequency': sp_freq,
                     'Oscillations': sp_osc,
                     'Symmetry': sp_sym,
                     # 'SOPhase': sp_cou,
                     'Stage': sp_sta}

        df_chan = pd.DataFrame(sp_params)

        # ####################################################################
        # END SINGLE CHANNEL DETECTION
        # ####################################################################
        df_chan['Channel'] = ch_names[i]
        df_chan['IdxChannel'] = i
        df = df.append(df_chan, ignore_index=True)

    # If no spindles were detected, return None
    if df.empty:
        logger.warning('No spindles were found in data. Returning None.')
        return None

    # Remove useless columns
    to_drop = []
    if hypno is None:
        to_drop.append('Stage')
    else:
        df['Stage'] = df['Stage'].astype(int)
    if len(to_drop):
        df = df.drop(columns=to_drop)

    # Find spindles that are present on at least two channels
    if multi_only and df['Channel'].nunique() > 1:
        # We round to the nearest second
        idx_good = np.logical_or(df['Start'].round(0).duplicated(keep=False),
                                 df['End'].round(0).duplicated(keep=False)).to_list()
        df = df[idx_good].reset_index(drop=True)

    return SpindlesResults(events=df,
                           data=data,
                           sf=sf,
                           ch_names=ch_names,
                           hypno=hypno,
                           data_filt=data_sigma,
                           data_broad=data_broad)


# %%
class SpindlesResults(_DetectionResults):
    """Output class for spindles detection.

    Attributes
    ----------
    _events : :py:class:`pandas.DataFrame`
        Output detection dataframe
    _data : array_like
        Original EEG data of shape *(n_chan, n_samples)*.
    _data_filt : array_like
        Sigma-filtered EEG data of shape *(n_chan, n_samples)*.
    _sf : float
        Sampling frequency of data.
    _ch_names : list
        Channel names.
    _hypno : array_like or None
        Sleep staging vector.
    """

    def __init__(self, events, data, sf, ch_names, hypno, data_filt, data_broad):
        self._data_broad = data_broad
        super().__init__(events, data, sf, ch_names, hypno, data_filt)

    def summary(self, grp_chan=False, grp_stage=False, aggfunc='mean',
                sort=True):
        """Return a summary of the spindles detection, optionally grouped
        across channels and/or stage.

        Parameters
        ----------
        grp_chan : bool
            If True, group by channel (for multi-channels detection only).
        grp_stage : bool
            If True, group by sleep stage (provided that an hypnogram was
            used).
        aggfunc : str or function
            Averaging function (e.g. ``'mean'`` or ``'median'``).
        sort : bool
            If True, sort group keys when grouping.
        """
        return super().summary(event_type='spindles',
                               grp_chan=grp_chan, grp_stage=grp_stage,
                               aggfunc=aggfunc, sort=sort)

    def get_mask(self):
        """Return a boolean array indicating for each sample in data if this
        sample is part of a detected event (True) or not (False).
        """
        return super().get_mask()

    def get_sync_events(self, center='Peak', time_before=1, time_after=1,
                        filt=(None, None)):
        """
        Return the raw or filtered data of each detected event after
        centering to a specific timepoint.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on.
            Default is to use the center peak of the spindles.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(1, 30)``
            will apply a 1 to 30 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using default
            parameters in the :py:func:`mne.filter.filter_data` function.

        Returns
        -------
        df_sync : :py:class:`pandas.DataFrame`
            Long-format dataframe::

            'Event' : Event number
            'Time' : Timing of the events (in seconds)
            'Amplitude' : Raw or filtered data for event
            'Channel' : Channel
            'IdxChannel' : Index of channel in data
            'Stage': Sleep stage in which the events occured (if available)
        """
        return super().get_sync_events(center=center, time_before=time_before,
                                       time_after=time_after, filt=filt)

    def plot_average(self, center='Peak', hue='Channel', time_before=1, time_after=1,
                     filt=(None, None), figsize=(6, 4.5), **kwargs):
        """
        Plot the average spindle.

        Parameters
        ----------
        center : str
            Landmark of the event to synchronize the timing on.
            Default is to use the most prominent peak of the spindle.
        hue : str
            Grouping variable that will produce lines with different colors.
            Can be either 'Channel' or 'Stage'.
        time_before : float
            Time (in seconds) before ``center``.
        time_after : float
            Time (in seconds) after ``center``.
        filt : tuple
            Optional filtering to apply to data. For instance, ``filt=(1, 30)``
            will apply a 1 to 30 Hz bandpass filter, and ``filt=(None, 40)``
            will apply a 40 Hz lowpass filter. Filtering is done using default
            parameters in the :py:func:`mne.filter.filter_data` function.
        figsize : tuple
            Figure size in inches.
        **kwargs : dict
            Optional argument that are passed to :py:func:`seaborn.lineplot`.
        """
        return super().plot_average(event_type='spindles', center=center,
                                    hue=hue, time_before=time_before,
                                    time_after=time_after, filt=filt,
                                    figsize=figsize, **kwargs)

    def plot_detection(self):
        """Plot an overlay of the detected spindles on the EEG signal.

        This only works in Jupyter and it requires the ipywidgets
        (https://ipywidgets.readthedocs.io/en/latest/) package.

        To activate the interactive mode, make sure to run:

        >>> %matplotlib widget

        .. versionadded:: 0.4.0
        """
        return super().plot_detection()

# %% [markdown]
# # Examples

# %%
sf = 1e3 / 0.01

# %% [markdown]
# ## Data 1

# %%
if __name__ == '__main__':
    data_file = 'exp_results/hp3/r1,NaK_th=-50,IT_th=-3,b=0.3,rho_p=0.50,E_KL=-100,g_L=0.050,E_L=-60/'
    data_file += 'gjw=0.002,freq=100/g_max=0.000200/E=-60.0/Vr=-64.00.npz'
    data = np.load(data_file)

    lfp1 = data['lfp']

# %% [markdown]
# ### Filter = 7-15 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp1, sf=1e3 / 0.01, show=True, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 0.5},
                          corr_window=0.3, corr_step=0.1, freq_sp=(7, 15))
    if sps:
        display(sps.summary())

# %% [markdown]
# ### Filter = 10-15 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp1, sf=1e3 / 0.01, show=True, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 0.5},
                          corr_window=0.3, corr_step=0.1, freq_sp=(10, 15))
    if sps:
        display(sps.summary())

# %% [markdown]
# ## Data 2

# %%
if __name__ == '__main__':
    data_file = 'exp_results/hp3/r1,NaK_th=-50,IT_th=-3,b=0.3,rho_p=0.50,E_KL=-100,g_L=0.050,E_L=-60/'
    data_file += 'gjw=0.002,freq=100/g_max=0.000200/E=-60.0/Vr=-66.00.npz'
    data = np.load(data_file)

    lfp2 = data['lfp']

# %% [markdown]
# ### Filter = 10-15 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp2, sf=1e3 / 0.01, show=True, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 0.5},
                          corr_window=0.3, corr_step=0.1, freq_sp=(10, 15))
    if sps:
        display(sps.summary())

# %% [markdown]
# ### Filter = 6 - 10 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp2, sf=sf, show=True, thresh={'rel_pow': 0.9, 'corr': 0.9, 'rms': 0.05},
                          corr_window=0.1, corr_step=0.05, freq_sp=(6, 10))
    if sps:
        display(sps.summary())

# %% [markdown]
# ## Data 3

# %%
if __name__ == '__main__':
    data_file = 'exp_results/hp3/r1,NaK_th=-50,IT_th=-3,b=0.3,rho_p=0.50,E_KL=-100,g_L=0.050,E_L=-60/'
    data_file += 'gjw=0.002,freq=100/g_max=0.000200/E=-60.0/Vr=-67.50.npz'
    data = np.load(data_file)
    lfp3 = data['lfp']

# %% [markdown]
# ### Filter = 10-15 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp3, sf=1e3 / 0.01, show=True, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 0.5},
                          corr_window=0.3, corr_step=0.1, freq_sp=(10, 15))
    if sps:
        display(sps.summary())

# %% [markdown]
# ## Data 4

# %%
if __name__ == '__main__':
    data_file = 'exp_results/hypothesis2/gaussian,mu=2.5,sigma=0.8,in=1.2_3.8,seed=57362/'
    data_file += 'r1,NaK_th=-50,IT_th=-3,b=0.3,rho_p=0.50,E_KL=-100,g_L=0.050,E_L=-60/'
    data_file += 'gjw=0.002,freq=100/g_max=0.0001,E=-65,Vr=-64.5-2.npz'
    data = np.load(data_file)

    lfp4 = data['V']

# %% [markdown]
# ### Filter = 10-15 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp4, sf=sf, show=True, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 0.5},
                          corr_window=0.3, corr_step=0.1, freq_sp=(10, 15))
    if sps:
        display(sps.summary())

# %% [markdown]
# ### Filter = 6-10 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp4, sf=sf, show=True, thresh={'rel_pow': 0.9, 'corr': 0.9, 'rms': 0.05},
                          corr_window=0.1, corr_step=0.05, freq_sp=(6, 10), sp_threshold=2.1)
    if sps:
        display(sps.summary())

# %% [markdown]
# ## Data 5

# %%
if __name__ == '__main__':
    #data_file = 'exp_results/hypothesis2/gaussian,mu=2.5,sigma=0.8,in=1.2_3.8,seed=57362/'
    #data_file += 'r1,NaK_th=-50,IT_th=-3,b=0.3,rho_p=0.50,E_KL=-100,g_L=0.050,E_L=-60/'
    #data_file += 'gjw=0.001,freq=100/g_max=0.0001,E=-65,Vr=-65.npz'
    data_file = 'exp_results/g_max=0.0001,E=-65,Vr=-65.npz'
    data = np.load(data_file)

    lfp5 = np.mean(data['V'], axis=1)

# %% [markdown]
# ### Filter = 10-15 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp5, sf=sf, show=True, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 0.5}, 
                          corr_window=0.3, corr_step=0.1, freq_sp=(10, 15))
    if sps:
        display(sps.summary())

# %% [markdown]
# ### Filter = 6-10 Hz

# %%
if __name__ == '__main__':
    sps = spindles_detect(lfp5, sf=sf, show=True, thresh={'rel_pow': 0.9, 'corr': 0.9, 'rms': 0.05}, 
                      corr_window=0.1, corr_step=0.05, freq_sp=(6, 10))
    if sps:
        display(sps.summary())


# %%

# %%

# %%
# # %%
# start, end = int(sps._events['Start'].loc[0] * sf), int(sps._events['End'].loc[0] * sf)
# plt.plot(sps._data[0, start: end])
#
# # %%
# start, end = int(sps._events['Start'].loc[1] * sf), int(sps._events['End'].loc[1] * sf)
# plt.plot(sps._data[0, start: end])
#
# # %%
# start, end = int(sps._events['Start'].loc[2] * sf), int(sps._events['End'].loc[2] * sf)
# plt.plot(sps._data[0, start: end])
#
# # %%
# sps.plot_detection()
#
# # %%
# sps.plot_average(center='Peak', filt=(12, 16), ci=None)
