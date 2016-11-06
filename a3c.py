import keras.backend as K
import numpy as np
from keras.models import Model

from multiprocessing import Pool, cpu_count
import threading as z


class A3C:
    def __init__(self, p, v, recurrent=False, continuous=False):
        """
        Reinforcement learning, asynchronous advantage actor-critic

        p: keras NN of policy
        v: keras NN of value prediction.
        recurrent: boolean, this requires dim of inputs and outputs has
            timestep's dim, like (n_batch, n_timestep, ...).
        continuous: boolean, continuous actions or not
        """

        assert (not self.recurrent) or (p.stateful and v.stateful)

        p_has_lock = hasattr(p, "lock")
        v_has_lock = hasattr(v, "lock")
        if p_has_lock and not v_has_lock:
            v.lock = p.lock
        elif v_has_lock and not p_has_lock:
            p.lock = v.lock
        elif p_has_lock and v_has_lock:
            if p.lock is not v.lock:
                raise Exception("p.lock is not v.lock")
        else:
            p.lock = v.lock = z.Lock()

        self.lock = p.lock
        self.p = p
        self.v = v
        self.recurrent = recurrent
        self.continuous = continuous
        self.T = 0
        self.pool = Pool(cpu_count())

    def compile(self, optimizer, beta=1.0):
        """
        compiles keras models, with keras optimizer.

        `optimizer`: keras optimizer
        `beta`: control parameter of entropy.
        """

        def loss_p(diff_r_v, a):
            c = self.continuous
            term = a if c else K.log(a)
            p, v, pi = K.sum(a, axis=-2)/K.sum(a), K.var(a), np.pi
            entropy = -(K.log(2*pi*v)+1)/2 if c else -K.sum(p*K.log(p))
            return K.mean(term * diff_r_v, axis=-1) + beta * entropy

        self.p.compile(optimizer=optimizer, loss=loss_p)
        self.v.compile(optimizer=optimizer, loss='mse')

    def one_episode(self, env, gamma, t_max, **kargs):
        # clone models for async training
        p = Model.from_config(self.p.get_config())
        v = Model.from_config(self.v.get_config())
        p_w0 = self.p.get_weights()
        v_w0 = self.v.get_weights()
        p.set_weights(p_w0)
        v.set_weights(v_w0)
        p.reset_states()
        v.reset_states()

        # loop of states->acts, `env.step` should be compatible with keras'
        # inputs.
        h_s, h_r, h_g = [], [], []
        t = 0
        done = False
        s = env.reset()
        while (not done) or (t < t_max):
            a = p.predict(np.reshape(s, (1, 1, -1)) if self.recurrent
                          else np.reshape(s, (1, -1)))
            a = a[-1, -1] if self.recurrent else a[-1]
            s_next, r, done, info = env.step(
                a if self.continuous else np.argmax(a, axis=-1))
            # `g` generalized gamma which makes bellman's equaltion applies to
            # hetergenous time delay.
            g = gamma ** (info['time'] if 'time' in info.keys() else 1)
            h_s.append(s)
            h_r.append(r)
            h_g.append(g)
            s = s_next
            t += 1
        self.T += t

        R = 0 if done else v.predict(
            np.reshape(s, (1, 1, -1)) if self.recurrent
            else np.reshape(s, (1, -1))).flatten()[-1]

        # summing gradients
        h_R = []
        for g, r in zip(h_g[::-1], h_r[::-1]):
            h_R.append(r + g * R)
        h_R = h_R[::-1]

        ss = np.vstack(h_s)
        RR = np.vstack(h_R)
        if self.recurrent:
            ss = np.expand_dims(ss, axis=0)
            RR = np.expand_dims(RR, axis=0)

        vv = v.predict(ss)
        diff_RR = RR - vv

        # parallel training with lock
        def fff(nn, target):
            nn.fit(ss, target, **kargs)
        self.pool.map_async(
            fff, [(p, diff_RR),
                  (v, RR)])

        # update_weights with lock
        def ggg(nn, nn_w0, new_nn):
            nn_w = nn.get_weights()
            new_nn_w = new_nn.get_weights()
            assert not (nn_w0[0] is new_nn_w[0])
            nn.set_weights([
                w + (w1 - w0) for w, w0, w1 in
                zip(nn_w, nn_w0, new_nn_w)])
        with self.lock:
            self.pool.map_async(
                ggg, [(self.p, p_w0, p),
                      (self.v, v_w0, v)])

    def train_env(self, T_max, env, gamma=0.9, t_max=np.inf, **kargs):
        """
        Training A3C

        T_max: max num of episodes.
        env: similiar to openai gym but only `env.step` method has
            to be implemeted. The observations and actions of `env.step` must
            be compitable with keras inputs and outputs.
        gamma: discount factor of rewards per unit time delay between
            current and next observations.
        t_max: max num of steps per episode.
        **kargs: pass extra parameters to keras model.fit function.
        """
        # async training steps
        def fff(i):
            while self.T < T_max:
                self.one_episode(
                    env=env, gamma=gamma, t_max=t_max, **kargs)
        self.pool.map_async(fff, range(cpu_count()))
