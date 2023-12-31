# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt


class TRNNeuronModel(bp.NeuGroup):
  def __init__(self, size, name=None, T=36.):
    super(TRNNeuronModel, self).__init__(size=size, name=name)

    self.IT_th = -3.
    self.b = 0.5
    self.g_T = 2.0
    self.g_L = 0.02
    self.E_L = -70.
    self.g_KL = 0.005
    self.E_KL = -95.
    self.NaK_th = -55.

    # temperature
    self.T = T

    # parameters of INa, IK
    self.g_Na = 100.
    self.E_Na = 50.
    self.g_K = 10.
    self.phi_m = self.phi_h = self.phi_n = 3 ** ((self.T - 36) / 10)

    # parameters of IT
    self.E_T = 120.
    self.phi_p = 5 ** ((self.T - 24) / 10)
    self.phi_q = 3 ** ((self.T - 24) / 10)
    self.p_half, self.p_k = -52., 7.4
    self.q_half, self.q_k = -80., -5.

    # parameters of V
    self.C, self.Vth, self.area = 1., 20., 1.43e-4
    self.V_factor = 1e-3 / self.area


class OriginalTRN(TRNNeuronModel):
  """TRN neuron model which is inspired from [1, 2].

  References
  -------
  [1] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski,
      T. J. (1999). Self–sustained rhythmic activity in the
      thalamic reticular nucleus mediated by depolarizing
      GABA A receptor potentials. Nature neuroscience, 2(2),
      168-174.
  [2] Bazhenov, Maxim, Igor Timofeev, Mircea Steriade, and
      Terrence J. Sejnowski. "Cellular and network models for
      intrathalamic augmenting responses during 10-Hz stimulation."
      Journal of Neurophysiology 79, no. 5 (1998): 2730-2748.

  """

  def __init__(self, size, **kwargs):
    super(OriginalTRN, self).__init__(size=size, **kwargs)

    self.V = bm.Variable(bm.zeros(self.num))
    self.m = bm.Variable(bm.zeros(self.num))
    self.h = bm.Variable(bm.zeros(self.num))
    self.n = bm.Variable(bm.zeros(self.num))
    self.p = bm.Variable(bm.zeros(self.num))
    self.q = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.input = bm.Variable(bm.zeros(self.num))

    self.int_m = bp.ode.ExponentialEuler(self.fm)
    self.int_h = bp.ode.ExponentialEuler(self.fh)
    self.int_n = bp.ode.ExponentialEuler(self.fn)
    self.int_p = bp.ode.ExponentialEuler(self.fp)
    self.int_q = bp.ode.ExponentialEuler(self.fq)
    self.int_V = bp.ode.ExponentialEuler(self.fV)

  def fm(self, m, t, V):
    alpha = 0.32 * (V - self.NaK_th - 13.) / (1 - bm.exp(-(V - self.NaK_th - 13.) / 4.))
    beta = -0.28 * (V - self.NaK_th - 40.) / (1 - bm.exp((V - self.NaK_th - 40.) / 5.))
    tau = 1. / self.phi_m / (alpha + beta)
    inf = alpha / (alpha + beta)
    dmdt = (inf - m) / tau
    return dmdt

  def fh(self, h, t, V):
    alpha = 0.128 * bm.exp(-(V - self.NaK_th - 17.) / 18.)
    beta = 4. / (1. + bm.exp(-(V - self.NaK_th - 40.) / 5.))
    tau = 1. / self.phi_h / (alpha + beta)
    inf = alpha / (alpha + beta)
    dhdt = (inf - h) / tau
    return dhdt

  def fn(self, n, t, V):
    alpha = 0.032 * (V - self.NaK_th - 15.) / (1. - bm.exp(-(V - self.NaK_th - 15.) / 5.))
    beta = self.b * bm.exp(-(V - self.NaK_th - 10.) / 40.)
    tau = 1 / self.phi_n / (alpha + beta)
    inf = alpha / (alpha + beta)
    dndt = (inf - n) / tau
    return dndt

  def fp(self, p, t, V):
    inf = 1. / (1. + bm.exp((-V + self.p_half + self.IT_th) / self.p_k))
    tau = 3. + 1. / (bm.exp((V + 27. - self.IT_th) / 10.) +
                     bm.exp(-(V + 102. - self.IT_th) / 15.))
    dpdt = self.phi_p * (inf - p) / tau
    return dpdt

  def fq(self, q, t, V):
    inf = 1. / (1. + bm.exp(-(V - self.q_half - self.IT_th) / self.q_k))
    tau = 85. + 1. / (bm.exp((V + 48. - self.IT_th) / 4.) +
                      bm.exp(-(V + 407. - self.IT_th) / 50.))
    dqdt = self.phi_q * (inf - q) / tau
    return dqdt

  def fV(self, V, t, m, h, n, p, q, Isyn):
    INa = self.g_Na * m ** 3 * h * (V - self.E_Na)
    IK = self.g_K * n ** 4 * (V - self.E_KL)
    IT = self.g_T * p ** 2 * q * (V - self.E_T)
    IL = self.g_L * (V - self.E_L)
    IKL = self.g_KL * (V - self.E_KL)
    Icur = INa + IK + IT + IL + IKL
    dvdt = (-Icur + Isyn * self.V_factor) / self.C
    return dvdt

  def update(self, tdi, x=None):
    if x is not None:
      self.input += x
    t, dt = tdi['t'], tdi['dt']
    self.m.value = self.int_m(self.m, t, self.V, dt=dt)
    self.h.value = self.int_h(self.h, t, self.V, dt=dt)
    self.n.value = self.int_n(self.n, t, self.V, dt=dt)
    self.p.value = self.int_p(self.p, t, self.V, dt=dt)
    self.q.value = self.int_q(self.q, t, self.V, dt=dt)
    V = self.int_V(self.V, t, m=self.m, h=self.h, n=self.n, p=self.p, q=self.q, Isyn=self.input, dt=dt)
    self.spike.value = bm.logical_and(V >= self.Vth, self.V < self.Vth)
    self.V.value = V
    self.input[:] = 0.

  def reset(self, Vr):
    self.V[:] = Vr

    alpha = 0.32 * (self.V - self.NaK_th - 13.) / (1 - bm.exp(-(self.V - self.NaK_th - 13.) / 4.))
    beta = -0.28 * (self.V - self.NaK_th - 40.) / (1 - bm.exp((self.V - self.NaK_th - 40.) / 5.))
    self.m[:] = alpha / (alpha + beta)

    alpha = 0.128 * bm.exp(-(self.V - self.NaK_th - 17.) / 18.)
    beta = 4. / (1. + bm.exp(-(self.V - self.NaK_th - 40.) / 5.))
    self.h[:] = alpha / (alpha + beta)

    alpha = 0.032 * (self.V - self.NaK_th - 15.) / (1. - bm.exp(-(self.V - self.NaK_th - 15.) / 5.))
    beta = self.b * bm.exp(-(self.V - self.NaK_th - 10.) / 40.)
    self.n[:] = alpha / (alpha + beta)

    self.p[:] = 1. / (1. + bm.exp((-self.V - 52. + self.IT_th) / 7.4))
    self.q[:] = 1. / (1. + bm.exp((self.V + 80. - self.IT_th) / 5.))

    self.spike[:] = False
    self.input[:] = 0.


class ReducedTRN(TRNNeuronModel):
  def __init__(self, size, **kwargs):
    super(ReducedTRN, self).__init__(size=size, **kwargs)

    # parameters
    self.b = 0.14
    self.rho_p = 0.

    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.y = bm.Variable(bm.zeros(self.num))
    self.z = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.input = bm.Variable(bm.zeros(self.num))

  def get_channel_currents(self, V, y, z, g_T):
    # m channel
    t1 = 13. - V + self.NaK_th
    t1_exp = bm.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    t2 = V - 40. - self.NaK_th
    t2_exp = bm.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)

    # h channel
    h_alpha_by_y = 0.128 * bm.exp((17. - y + self.NaK_th) / 18.)  # \alpha_h(y)
    t3 = bm.exp((40. - y + self.NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # \h_{\infty}(y)

    # n channel
    t4 = 15. - y + self.NaK_th
    n_alpha_by_y = 0.032 * t4 / (bm.exp(t4 / 5.) - 1.)  # \alpha_n(y)
    n_beta_by_y = self.b * bm.exp((10. - y + self.NaK_th) / 40.)  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_by_y)  # n_{\infty}(y)

    # p channel
    p_inf_by_y = 1. / (1. + bm.exp((self.p_half - y + self.IT_th) / self.p_k))  # p_{\infty}(y)
    q_inf_by_z = 1. / (1. + bm.exp((self.q_half - z + self.IT_th) / self.q_k))  # q_{\infty}(z)

    # currents
    gK = self.g_K * n_inf_by_y ** 4  # gK
    gNa = self.g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
    gT = g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
    INa = gNa * (V - self.E_Na)
    IK = gK * (V - self.E_KL)
    IT = gT * (V - self.E_T)
    return INa, IK, IT

  def get_INaIK_currents(self, V, y, z):
    # m channel
    t1 = 13. - V + self.NaK_th
    t1_exp = bm.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    t2 = V - 40. - self.NaK_th
    t2_exp = bm.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)

    # h channel
    h_alpha_by_y = 0.128 * bm.exp((17. - y + self.NaK_th) / 18.)  # \alpha_h(y)
    t3 = bm.exp((40. - y + self.NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # \h_{\infty}(y)

    # n channel
    t4 = 15. - y + self.NaK_th
    n_alpha_by_y = 0.032 * t4 / (bm.exp(t4 / 5.) - 1.)  # \alpha_n(y)
    n_beta_by_y = self.b * bm.exp((10. - y + self.NaK_th) / 40.)  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_by_y)  # n_{\infty}(y)

    # p channel
    p_inf_by_y = 1. / (1. + bm.exp((self.p_half - y + self.IT_th) / self.p_k))  # p_{\infty}(y)
    q_inf_by_z = 1. / (1. + bm.exp((self.q_half - z + self.IT_th) / self.q_k))  # q_{\infty}(z)

    # currents
    gK = self.g_K * n_inf_by_y ** 4  # gK
    gNa = self.g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
    gT_before = p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
    INa = gNa * (V - self.E_Na)
    IK = gK * (V - self.E_KL)
    # IT = gT * (V - self.E_T)
    return INa, IK, gT_before

  def f_optimize_v(self, V, g_T, Iext, g_L, E_L, g_KL, E_KL):
    INa, IK, IT = self.get_channel_currents(V, y=V, z=V, g_T=g_T)
    IL = g_L * (V - E_L)
    IKL = g_KL * (V - E_KL)
    dxdt = -INa - IK - IT - IL - IKL + self.V_factor * Iext
    return dxdt

  def get_resting_potential(self, g_T, Iext, g_L, E_L, g_KL, E_KL):
    vs = bm.arange(-100, 55, 0.01)
    roots = bp.analysis.utils.roots_of_1d_by_x(bm.jit(self.f_optimize_v), vs,
                                               args=(g_T, Iext, g_L, E_L, g_KL, E_KL))
    return roots

  def suggest_gL(self, Vr, g_T, Iext, E_L, g_KL, E_KL):
    INa, IK, IT = self.get_channel_currents(V=Vr, y=Vr, z=Vr, g_T=g_T)
    IKL = g_KL * (Vr - E_KL)
    gL = (-INa - IK - IT - IKL + Iext) / (Vr - E_L)
    return gL

  def suggest_gKL(self, Vr, g_T, Iext):
    INa, IK, IT = self.get_channel_currents(V=Vr, y=Vr, z=Vr, g_T=g_T)
    IL = self.g_L * (Vr - self.E_L)
    gKL = (-INa - IK - IT - IL + Iext) / (Vr - self.E_KL)
    return bm.asarray(gKL)

  def suggest_gT(self, Vr, g_KL, Iext):  ########################################
    INa, IK, gT_before = self.get_INaIK_currents(V=Vr, y=Vr, z=Vr)
    IL = self.g_L * (Vr - self.E_L)
    IKL = g_KL * (Vr - self.E_KL)
    g_T = (-INa - IK - IKL - IL + Iext) / (Vr - self.E_T) / gT_before
    return bm.asarray(g_T)

  def reset_state(self, Vr):
    self.V[:] = Vr
    self.y[:] = Vr
    self.z[:] = Vr
    self.spike[:] = False
    self.input[:] = 0.


class ReducedTRNv1(ReducedTRN):
  """Reduced TRN version 1.

  In this reduced TRN neuron model, we make two reductions:

  1. group n, h, p channels.
  2. group V, and m channel.

  """

  def fV(self, V, t, y, z, Isyn):
    # m channel
    t1 = 13. - V + self.NaK_th
    t1_exp = bm.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    m_alpha_by_V_diff = (-0.32 * (t1_exp - 1.) + 0.08 * t1 * t1_exp) / (t1_exp - 1.) ** 2  # \alpha_m'(V)
    t2 = V - 40. - self.NaK_th
    t2_exp = bm.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_beta_by_V_diff = (0.28 * (t2_exp - 1) - 0.056 * t2 * t2_exp) / (t2_exp - 1) ** 2  # \beta_m'(V)
    m_tau_by_V = 1. / self.phi_m / (m_alpha_by_V + m_beta_by_V)  # \tau_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)
    m_inf_by_V_diff = (m_alpha_by_V_diff * m_beta_by_V - m_alpha_by_V * m_beta_by_V_diff) / \
                      (m_alpha_by_V + m_beta_by_V) ** 2  # \m_{\infty}'(V)

    # h channel
    h_alpha_by_y = 0.128 * bm.exp((17. - y + self.NaK_th) / 18.)  # \alpha_h(y)
    t3 = bm.exp((40. - y + self.NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)

    # n channel
    t5 = (15. - y + self.NaK_th)
    t5_exp = bm.exp(t5 / 5.)
    n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
    t6 = bm.exp((10. - y + self.NaK_th) / 40.)
    n_beta_y = self.b * t6  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)

    # p channel
    t7 = bm.exp((self.p_half - y + self.IT_th) / self.p_k)
    p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
    t8 = bm.exp((self.q_half - z + self.IT_th) / self.q_k)
    q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)

    # x
    gNa = self.g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
    gK = self.g_K * n_inf_by_y ** 4  # gK
    gT = self.g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
    FV = gNa + gK + gT + self.g_L + self.g_KL  # dF/dV
    Fm = 3 * self.g_Na * h_inf_by_y * (V - self.E_Na) * m_inf_by_V * m_inf_by_V * m_inf_by_V_diff  # dF/dvm
    t9 = self.C / m_tau_by_V
    t10 = FV + Fm
    t11 = t9 + FV
    rho_V = (t11 - bm.sqrt(bm.maximum(t11 ** 2 - 4 * t9 * t10, 0.))) / 2 / t10  # rho_V
    INa = gNa * (V - self.E_Na)
    IK = gK * (V - self.E_KL)
    IT = gT * (V - self.E_T)
    IL = self.g_L * (V - self.E_L)
    IKL = self.g_KL * (V - self.E_KL)
    Iext = self.V_factor * Isyn
    dVdt = rho_V * (-INa - IK - IT - IL - IKL + Iext) / self.C

    return dVdt

  def fy(self, y, t, V):
    # m channel
    t1 = 13. - V + self.NaK_th
    t1_exp = bm.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    t2 = V - 40. - self.NaK_th
    t2_exp = bm.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)

    # h channel
    h_alpha_by_V = 0.128 * bm.exp((17. - V + self.NaK_th) / 18.)  # \alpha_h(V)
    h_beta_by_V = 4. / (bm.exp((40. - V + self.NaK_th) / 5.) + 1.)  # \beta_h(V)
    h_inf_by_V = h_alpha_by_V / (h_alpha_by_V + h_beta_by_V)  # h_{\infty}(V)
    h_tau_by_V = 1. / self.phi_h / (h_alpha_by_V + h_beta_by_V)  # \tau_h(V)
    h_alpha_by_y = 0.128 * bm.exp((17. - y + self.NaK_th) / 18.)  # \alpha_h(y)
    t3 = bm.exp((40. - y + self.NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_beta_by_y_diff = 0.8 * t3 / (1 + t3) ** 2  # \beta_h'(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)
    h_alpha_by_y_diff = - h_alpha_by_y / 18.  # \alpha_h'(y)
    h_inf_by_y_diff = (h_alpha_by_y_diff * h_beta_by_y - h_alpha_by_y * h_beta_by_y_diff) / \
                      (h_beta_by_y + h_alpha_by_y) ** 2  # h_{\infty}'(y)

    # n channel
    t4 = (15. - V + self.NaK_th)
    n_alpha_by_V = 0.032 * t4 / (bm.exp(t4 / 5.) - 1.)  # \alpha_n(V)
    n_beta_by_V = self.b * bm.exp((10. - V + self.NaK_th) / 40.)  # \beta_n(V)
    n_tau_by_V = 1. / (n_alpha_by_V + n_beta_by_V) / self.phi_n  # \tau_n(V)
    n_inf_by_V = n_alpha_by_V / (n_alpha_by_V + n_beta_by_V)  # n_{\infty}(V)
    t5 = (15. - y + self.NaK_th)
    t5_exp = bm.exp(t5 / 5.)
    n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
    t6 = bm.exp((10. - y + self.NaK_th) / 40.)
    n_beta_y = self.b * t6  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)
    n_alpha_by_y_diff = (0.0064 * t5 * t5_exp - 0.032 * (t5_exp - 1.)) / (t5_exp - 1.) ** 2  # \alpha_n'(y)
    n_beta_by_y_diff = -n_beta_y / 40  # \beta_n'(y)
    n_inf_by_y_diff = (n_alpha_by_y_diff * n_beta_y - n_alpha_by_y * n_beta_by_y_diff) / \
                      (n_alpha_by_y + n_beta_y) ** 2  # n_{\infty}'(y)

    # p channel
    p_inf_by_V = 1. / (1. + bm.exp((self.p_half - V + self.IT_th) / self.p_k))  # p_{\infty}(V)
    p_tau_by_V = (3 + 1. / (bm.exp((V + 27. - self.IT_th) / 10.) +
                            bm.exp(-(V + 102. - self.IT_th) / 15.))) / self.phi_p  # \tau_p(V)
    t7 = bm.exp((self.p_half - y + self.IT_th) / self.p_k)
    p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
    p_inf_by_y_diff = t7 / self.p_k / (1. + t7) ** 2  # p_{\infty}'(y)

    #  y
    Fvh = self.g_Na * m_inf_by_V ** 3 * (V - self.E_Na) * h_inf_by_y_diff  # dF/dvh
    Fvn = 4 * self.g_K * (V - self.E_KL) * n_inf_by_y ** 3 * n_inf_by_y_diff  # dF/dvn
    f4 = Fvh + Fvn
    rho_h = (1 - self.rho_p) * Fvh / f4
    rho_n = (1 - self.rho_p) * Fvn / f4
    fh = (h_inf_by_V - h_inf_by_y) / h_tau_by_V / h_inf_by_y_diff
    fn = (n_inf_by_V - n_inf_by_y) / n_tau_by_V / n_inf_by_y_diff
    fp = (p_inf_by_V - p_inf_by_y) / p_tau_by_V / p_inf_by_y_diff
    dydt = rho_h * fh + rho_n * fn + self.rho_p * fp

    return dydt

  def fz(self, z, t, V):
    q_inf_by_V = 1. / (1. + bm.exp((self.q_half - V + self.IT_th) / self.q_k))  # q_{\infty}(V)
    t8 = bm.exp((self.q_half - z + self.IT_th) / self.q_k)
    q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)
    q_inf_diff_z = t8 / self.q_k / (1. + t8) ** 2  # q_{\infty}'(z)
    q_tau_by_V = (85. + 1 / (bm.exp((V + 48. - self.IT_th) / 4.) +
                             bm.exp(-(V + 407. - self.IT_th) / 50.))) / self.phi_q  # \tau_q(V)
    dzdt = (q_inf_by_V - q_inf_by_z) / q_tau_by_V / q_inf_diff_z
    return dzdt


  @property
  def derivative(self):
    return bp.JointEq(self.fV, self.fy, self.fz)

  def __init__(self, size, method='rk4'):
    super(ReducedTRNv1, self).__init__(size)
    self.integral = bp.odeint(self.derivative, method=method)

  def update(self, tdi):
    t, dt = tdi.t, tdi.dt
    V, self.y.value, self.z.value = self.integral(self.V, self.y, self.z, t, self.input, dt)
    self.spike.value = bm.logical_and((self.V < self.Vth), (V >= self.Vth))
    self.V.value = V
    self.input[:] = 0.





class ReducedTRNv2(ReducedTRN):
  def derivative(self, V, y, z, t, Isyn):
    # m channel
    t1 = 13. - V + self.NaK_th
    t1_exp = bm.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    m_alpha_by_V_diff = (-0.32 * (t1_exp - 1.) + 0.08 * t1 * t1_exp) / (t1_exp - 1.) ** 2  # \alpha_m'(V)
    t2 = V - 40. - self.NaK_th
    t2_exp = bm.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_beta_by_V_diff = (0.28 * (t2_exp - 1) - 0.056 * t2 * t2_exp) / (t2_exp - 1) ** 2  # \beta_m'(V)
    m_tau_by_V = 1. / self.phi_m / (m_alpha_by_V + m_beta_by_V)  # \tau_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)
    m_inf_by_V_diff = (m_alpha_by_V_diff * m_beta_by_V - m_alpha_by_V * m_beta_by_V_diff) / \
                      (m_alpha_by_V + m_beta_by_V) ** 2  # \m_{\infty}'(V)

    # h channel
    h_alpha_by_V = 0.128 * bm.exp((17. - V + self.NaK_th) / 18.)  # \alpha_h(V)
    h_beta_by_V = 4. / (bm.exp((40. - V + self.NaK_th) / 5.) + 1.)  # \beta_h(V)
    h_inf_by_V = h_alpha_by_V / (h_alpha_by_V + h_beta_by_V)  # h_{\infty}(V)
    h_tau_by_V = 1. / self.phi_h / (h_alpha_by_V + h_beta_by_V)  # \tau_h(V)
    h_alpha_by_y = 0.128 * bm.exp((17. - y + self.NaK_th) / 18.)  # \alpha_h(y)
    t3 = bm.exp((40. - y + self.NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_beta_by_y_diff = 0.8 * t3 / (1 + t3) ** 2  # \beta_h'(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)
    h_alpha_by_y_diff = - h_alpha_by_y / 18.  # \alpha_h'(y)
    h_inf_by_y_diff = (h_alpha_by_y_diff * h_beta_by_y - h_alpha_by_y * h_beta_by_y_diff) / \
                      (h_beta_by_y + h_alpha_by_y) ** 2  # h_{\infty}'(y)

    # n channel
    t4 = (15. - V + self.NaK_th)
    n_alpha_by_V = 0.032 * t4 / (bm.exp(t4 / 5.) - 1.)  # \alpha_n(V)
    n_beta_by_V = self.b * bm.exp((10. - V + self.NaK_th) / 40.)  # \beta_n(V)
    n_tau_by_V = 1. / (n_alpha_by_V + n_beta_by_V) / self.phi_n  # \tau_n(V)
    n_inf_by_V = n_alpha_by_V / (n_alpha_by_V + n_beta_by_V)  # n_{\infty}(V)
    t5 = (15. - y + self.NaK_th)
    t5_exp = bm.exp(t5 / 5.)
    n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
    t6 = bm.exp((10. - y + self.NaK_th) / 40.)
    n_beta_y = self.b * t6  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)
    n_alpha_by_y_diff = (0.0064 * t5 * t5_exp - 0.032 * (t5_exp - 1.)) / (t5_exp - 1.) ** 2  # \alpha_n'(y)
    n_beta_by_y_diff = -n_beta_y / 40  # \beta_n'(y)
    n_inf_by_y_diff = (n_alpha_by_y_diff * n_beta_y - n_alpha_by_y * n_beta_by_y_diff) / \
                      (n_alpha_by_y + n_beta_y) ** 2  # n_{\infty}'(y)

    # p channel
    p_inf_by_V = 1. / (1. + bm.exp((self.p_half - V + self.IT_th) / self.p_k))  # p_{\infty}(V)
    p_tau_by_V = (3 + 1. / (bm.exp((V + 27. - self.IT_th) / 10.) +
                            bm.exp(-(V + 102. - self.IT_th) / 15.))) / self.phi_p  # \tau_p(V)
    t7 = bm.exp((self.p_half - y + self.IT_th) / self.p_k)
    p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
    p_inf_by_y_diff = t7 / self.p_k / (1. + t7) ** 2  # p_{\infty}'(y)

    # p channel
    q_inf_by_V = 1. / (1. + bm.exp((self.q_half - V + self.IT_th) / self.q_k))  # q_{\infty}(V)
    t8 = bm.exp((self.q_half - z + self.IT_th) / self.q_k)
    q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)
    q_inf_diff_z = t8 / self.q_k / (1. + t8) ** 2  # q_{\infty}'(z)
    q_tau_by_V = (85. + 1 / (bm.exp((V + 48. - self.IT_th) / 4.) +
                             bm.exp(-(V + 407. - self.IT_th) / 50.))) / self.phi_q  # \tau_q(V)

    # ----
    #  x
    # ----

    gNa = self.g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
    gK = self.g_K * n_inf_by_y ** 4  # gK
    gT = self.g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
    FV = gNa + gK + gT + self.g_L + self.g_KL  # dF/dV
    Fm = 3 * self.g_Na * h_inf_by_y * (V - self.E_Na) * m_inf_by_V * m_inf_by_V * m_inf_by_V_diff  # dF/dvm
    t9 = self.C / m_tau_by_V
    t10 = FV + Fm
    t11 = t9 + FV
    rho_V = (t11 - bm.sqrt(bm.maximum(t11 ** 2 - 4 * t9 * t10, 0.))) / 2 / t10  # rho_V
    INa = gNa * (V - self.E_Na)
    IK = gK * (V - self.E_KL)
    IT = gT * (V - self.E_T)
    IL = self.g_L * (V - self.E_L)
    IKL = self.g_KL * (V - self.E_KL)
    Iext = self.V_factor * Isyn
    dVdt = rho_V * (-INa - IK - IT - IL - IKL + Iext) / self.C

    # ----
    #  y
    # ----

    Fvh = self.g_Na * m_inf_by_V ** 3 * (V - self.E_Na) * h_inf_by_y_diff  # dF/dvh
    Fvn = 4 * self.g_K * (V - self.E_KL) * n_inf_by_y ** 3 * n_inf_by_y_diff  # dF/dvn
    f4 = Fvh + Fvn
    rho_h = (1 - self.rho_p) * Fvh / f4
    rho_n = (1 - self.rho_p) * Fvn / f4
    fh = (h_inf_by_V - h_inf_by_y) / h_tau_by_V / h_inf_by_y_diff
    fn = (n_inf_by_V - n_inf_by_y) / n_tau_by_V / n_inf_by_y_diff
    fp = (p_inf_by_V - p_inf_by_y) / p_tau_by_V / p_inf_by_y_diff
    dydt = rho_h * fh + rho_n * fn + self.rho_p * fp

    # ----
    #  z
    # ----

    dzdt = (q_inf_by_V - q_inf_by_z) / q_tau_by_V / q_inf_diff_z

    return dVdt, dydt, dzdt

  def __init__(self, size, method='rk4'):
    super(ReducedTRNv2, self).__init__(size)
    self.integral = bp.odeint(self.derivative, method=method)

  def update(self, tdi):
    t, dt = tdi.t, tdi.dt
    V, self.y.value, self.z.value = self.integral(self.V, self.y, self.z, t, self.input, dt)
    self.spike.value = bm.logical_and(self.V < self.Vth, V >= self.Vth)
    self.V.value = V
    self.input[:] = 0.


class GABAaOne2One(bp.TwoEndConn):
  def __init__(self, pre, post, conn=None, g_max=0.42, alpha=10.5,
               beta=0.18, T=1.0, T_duration=1.0, E=-80.,  # E=-62.
               method='rk4'):
    super(GABAaOne2One, self).__init__(pre=pre, post=post, conn=conn)

    # parameters
    self.beta = beta
    self.T = T
    self.T_duration = T_duration
    self.g_max = g_max
    self.alpha = alpha
    self.E = E

    # variables
    self.s = bm.Variable(bm.zeros(post.num))
    self.I = bm.Variable(bm.zeros(post.num))

    # function
    self.int_s = bp.odeint(lambda s, t, TT, alpha, beta: alpha * TT * (1 - s) - beta * s,
                           method=method)

  def update(self, tdi):
    t, dt = tdi.t, tdi.dt
    TT = ((t - self.pre.t_last_spike) < self.T_duration) * self.T
    self.s.value = self.int_s(self.s, t, TT, self.alpha, self.beta, dt)
    I = self.g_max * self.s * (self.post.V - self.E)
    self.I.value = I
    self.post.input -= I

  def reset(self):
    self.s[:] = 0.


# ====================Gapjunction================== #
def get_val(current):
  if np.max(current) != 0:
    return np.max(current)
  elif np.min(current) != 0:
    return np.min(current)
  else:
    return 0


# class GapJunction(bp.TwoEndConn):    

#   def __init__(self, group, conn, gjw, **kwargs):
#     super(GapJunction, self).__init__(pre=group, post=group, conn=conn, **kwargs)
#     self.gjw = gjw
#     self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')

#   def update(self, t, dt):
#     f = bm.vmap(lambda V, pre_i, post_i: V[pre_i] - V[post_i],
#                 in_axes=(None, 0, 0))
#     diff = f(self.pre.V, self.pre_ids, self.post_ids)
#     self.post.input += self.gjw * bm.syn2post(diff, self.post_ids, self.post.num)

#   def reset(self):
#     pass


class GapJunction(bp.TwoEndConn):
  def __init__(self, group, conn, gjw, **kwargs):
    super(GapJunction, self).__init__(pre=group, post=group, conn=conn, **kwargs)
    self.gjw = gjw
    conn_mat = self.conn.requires('conn_mat')
    self.conn_mat = bm.logical_and(conn_mat, conn_mat.T)

  def update(self, tdi):
    diff = bm.expand_dims(self.pre.V, 1) - self.post.V
    self.post.input += self.gjw * bm.sum(diff * self.conn_mat, 0)

  def reset(self):
    pass


class Poisson_Input(bp.NeuGroup):
  """Poisson Neuron Group.

  Parameters
  ----------
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, freqs, seed=None, **kwargs):
    super(Poisson_Input, self).__init__(size=size, **kwargs)

    self.freqs = freqs
    self.dt = bm.get_dt() / 1000.
    self.size = (size,) if isinstance(size, int) else tuple(size)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
    self.rng = bm.random.RandomState(seed=seed)

  def update(self, tdi):
    self.spike.value = self.rng.random(self.num) <= self.freqs * self.dt
    self.t_last_spike.value = bm.where(self.spike, tdi.t, self.t_last_spike)


class PoissonInput(Poisson_Input):
  def reset(self):
    self.t_last_spike[:] = -1e7
    self.spike[:] = False


### old version
# class TRNNet(bp.Network):
#   def __init__(self, size, method='rk4'):  #exp_auto   rk4
#     super(TRNNet, self).__init__()
#     self.P = PoissonInput(size, freqs=50.)
#     self.T = ReducedTRNv1(size, method=method)
#     self.P2T = GABAaOne2One(pre=self.P, post=self.T, alpha=1.,
#                             beta=0.18, T=0.5, T_duration=0.3,
#                             method=method)
#     self.GJ = GapJunction(group=self.T, conn=bp.connect.grid_four, gjw=0.001)

#   def update(self, t, dt):
#     self.GJ.update(t, dt)
#     self.P2T.update(t, dt)
#     self.P.update(t, dt)
#     self.T.update(t, dt)

#   def reset(self, Vr):
#     self.P.reset()
#     self.T.reset(Vr)
#     self.P2T.reset()
#     self.GJ.reset()

from brainpy import math


class RegularInput(bp.NeuGroup):
  """Regular Neuron Group.

  Parameters
  ----------
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, freqs, **kwargs):
    super().__init__(size=size, **kwargs)

    self.freqs = freqs
    self.dt = math.get_dt() / 1000.
    self.size = (size,) if isinstance(size, int) else tuple(size)
    self.spike = math.Variable(math.zeros(self.num, dtype=bool))
    self.t_last_spike = math.Variable(math.ones(self.num) * 0)

  def update(self, tdi):
    self.spike.value = (tdi.t - self.t_last_spike) >= (1 / self.freqs) * 1000
    # self.spike.value = self.rng.random(self.num) <= self.freqs * self.dt
    self.t_last_spike.value = math.where(self.spike, tdi.t, self.t_last_spike)

  def reset(self):
    self.t_last_spike[:] = 0
    self.spike[:] = False


# ---- standard TRN network ---- #   
class TRNNet(bp.Network):
  """Isolated TRN network model."""

  def __init__(self, size, conn, method='rk4'):
    super(TRNNet, self).__init__()
    self.P = PoissonInput(size, freqs=50.)  # RegularInput(size, freqs=50.)#PoissonInput(size, freqs=50.)
    self.T = ReducedTRNv1(size, method=method)
    self.P2T = GABAaOne2One(pre=self.P,
                            post=self.T,
                            alpha=1.,
                            beta=0.18,
                            T=0.5,
                            T_duration=0.3,
                            method=method)
    self.GJ = GapJunction(group=self.T, conn=conn, gjw=0.001)

  def update(self, tdi):
    self.GJ.update(tdi)
    self.P2T.update(tdi)
    self.P.update(tdi)
    self.T.update(tdi)

  def reset(self, Vr):
    self.P.reset()
    self.T.reset(Vr)
    self.P2T.reset()
    self.GJ.reset()


class TRNNet_ExternalAMPA(bp.Network):
  def __init__(self, size, conn, method='rk4', ampa_pars=dict()):
    super(TRNNet_ExternalAMPA, self).__init__()

    self.P = PoissonInput(size, freqs=50.)# bm.Variable(bm.zeros(1)))  # 50.
    self.T = ReducedTRNv1(size, method=method)
    self.P2T = GABAaOne2One(pre=self.P,
                            post=self.T,
                            alpha=1.,
                            beta=0.18,
                            T=0.5,
                            T_duration=0.3,
                            method='exp_auto')
    self.P2 = bp.neurons.PoissonGroup(size, freqs=50.) #bm.Variable(bm.zeros(1)))  #0.
    self.P2_to_T = bp.synapses.AMPA(self.P2,
                                    self.T,
                                    bp.conn.One2One(),
                                    method='exp_auto',
                                    **ampa_pars)
    self.GJ = GapJunction(group=self.T, conn=conn, gjw=0.001)

  def reset(self, Vr):
    self.P.reset()
    self.P2.reset()
    self.T.reset(Vr)
    self.P2T.reset()
    self.P2_to_T.reset()
    self.GJ.reset()

class External_Input(bp.NeuGroup):
  """Poisson Neuron Group.

  Parameters
  ----------
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, freqs, seed=None, **kwargs):
    super(External_Input, self).__init__(size=size, **kwargs)

    self.freqs = freqs
    self.dt = bm.get_dt() / 1000.
    self.size = (size,) if isinstance(size, int) else tuple(size)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
    self.rng = bm.random.RandomState(seed=seed)

  def update(self, tdi):
    self.spike.value = self.rng.random(self.num) <= self.freqs * self.dt
    self.t_last_spike.value = bm.where(self.spike, tdi.t, self.t_last_spike)


class ExternalInput(Poisson_Input):
  def reset(self):
    self.t_last_spike[:] = -1e7
    self.spike[:] = False
    
class TRNNet_ExternalInput(bp.Network):
  def __init__(self, size, conn, method='rk4', ampa_pars=dict()):
    super(TRNNet_ExternalInput, self).__init__()

    self.P = PoissonInput(size, freqs=50.)
    self.T = OriginalTRN(size, method=method)    #ReducedTRNv1(size, method=method)
    self.P2T = GABAaOne2One(pre=self.P,
                            post=self.T,
                            alpha=1.,
                            beta=0.18,
                            T=0.5,
                            T_duration=0.3,
                            method='exp_auto')
    self.P2 = bp.neurons.PoissonGroup(size, freqs=50.)
    self.P2_to_T = bp.synapses.AMPA(self.P2,
                                    self.T,
                                    bp.conn.One2One(),
                                    method='exp_auto',
                                    **ampa_pars)
    self.GJ = GapJunction(group=self.T, conn=conn, gjw=0.001)

  def reset(self, Vr):
    self.P.reset()
    self.P2.reset()
    self.T.reset(Vr)
    self.P2T.reset()
    self.P2_to_T.reset()
    self.GJ.reset()


# Synapse property
class GJNet(bp.Network):
  def __init__(self, conn, size, method='rk4'):  # exp_auto   rk4
    super(GJNet, self).__init__()
    self.T = ReducedTRNv1(size, method=method)
    self.GJ = GapJunction(group=self.T, conn=conn, gjw=0.01)

  def update(self, tdi):
    self.GJ.update(tdi)
    self.T.update(tdi)

  def reset(self, Vr):
    self.T.reset(Vr)
    self.GJ.reset()


# AMPA input, same gT, gjw vs. pop firing rate testing   
class AMPAOne2One(bp.TwoEndConn):
  def __init__(self, pre, post, conn=None, g_max=0.42, alpha=10.5,
               beta=0.18, T=1.0, T_duration=1.0, E=0.,  # T=0.5, T_duration=0.5
               method='rk4'):
    super(AMPAOne2One, self).__init__(pre=pre, post=post, conn=conn)

    # parameters
    self.beta = beta
    self.T = T
    self.T_duration = T_duration
    self.g_max = g_max
    self.alpha = alpha
    self.E = E

    # variables
    self.s = bm.Variable(bm.zeros(post.num))
    self.I = bm.Variable(bm.zeros(post.num))

    # function
    self.int_s = bp.odeint(lambda s, t, TT, alpha, beta: alpha * TT * (1 - s) - beta * s,
                           method=method)

  def update(self, tdi):
    t, dt = tdi.t, tdi.dt
    TT = ((t - self.pre.t_last_spike) < self.T_duration) * self.T
    self.s.value = self.int_s(self.s, t, TT, self.alpha, self.beta, dt=dt)
    I = self.g_max * self.s * (self.post.V - self.E)
    self.I.value = I
    self.post.input -= I

  def reset(self):
    self.s[:] = 0.


# AMPA input, same gT, gjw vs. pop firing rate testing 
class TRN_GJ_Net(bp.Network):
  """Isolated TRN network model."""

  def __init__(self, size, conn, method='rk4'):
    super(TRN_GJ_Net, self).__init__()
    self.P = PoissonInput(size, freqs=50.)
    self.T = ReducedTRNv1(size, method=method)
    self.P2T = AMPAOne2One(pre=self.P, post=self.T, alpha=1., beta=0.18,
                           T=0.5, T_duration=0.3, method=method)
    self.GJ = GapJunction(group=self.T, conn=conn, gjw=0.001)

  def update(self, tdi):
    self.GJ.update(tdi)
    self.P2T.update(tdi)
    self.P.update(tdi)
    self.T.update(tdi)

  def reset(self, Vr):
    self.P.reset()
    self.T.reset(Vr)
    self.P2T.reset()
    self.GJ.reset()


# ---- GABAa input to single neuron, test the property of GABAa synapse ----#

class GABAaNeuron(bp.Network):
  """Isolated TRN network model."""

  def __init__(self, size, method='rk4'):
    super(GABAaNeuron, self).__init__()
    self.P = RegularInput(size, freqs=50.)
    self.T = ReducedTRNv1(size, method=method)
    self.P2T = GABAaOne2One(pre=self.P, post=self.T, alpha=1., beta=0.18,
                            T=0.5, T_duration=0.3, method=method)
    # self.GJ = GapJunction(group=self.T, conn=conn, gjw=0.001)

  def update(self, tdi):
    # self.GJ.update(tdi)
    self.P2T.update(tdi)
    self.P.update(tdi)
    self.T.update(tdi)

  def reset(self, Vr):
    self.P.reset()
    self.T.reset(Vr)
    self.P2T.reset()
    # self.GJ.reset()


# ---- Gaussian gap junction connnection probiblity ----#
class GaussianProbForGJ(bp.conn.GaussianProb):
  """Gaussian Probability connection for Gap Junction."""

  def build_mat(self):
    conn_mat = super(GaussianProbForGJ, self).build_mat()
    conn_mat = np.logical_or(conn_mat, conn_mat.T)
    return conn_mat

  def visualize(self, i=0):
    _, mat = self.build_conn()  # self.data.toarray()
    plt.subplot(121)
    plt.imshow(mat)
    plt.title('Total connection matrix')
    plt.subplot(122)
    plt.imshow(mat[i].reshape(self.pre_size))
    plt.title(f'Connection with node {i}')
    plt.show()
