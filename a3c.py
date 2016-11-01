import keras.backend as K
import numpy as np
from keras.models import Model

from multiprocessing import Pool
from multiprocessing import cpu_count
import threading as z


def to_list(a):
    return a if type(a) is list else [a]


def de_list(a):
    return a[0] if type(a) is list and len(a) == 1 else a


class A3C:
    def __init__(self, p, v):
        """
        Reinforcement learning, asynchronous advantage actor-critic

        p: keras NN of policy
        v: keras NN of value prediction.
        """
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
        self.T = 0
        self.pool = Pool(cpu_count())

    def compile(self, optimizer):
        """compiles keras models, with keras optimizer"""

        def loss_p(diff_r_v, a):
            return K.mean(K.log(a) * diff_r_v, -1)

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

        # loop of states->acts, `env.step` should be compatible with keras'
        # inputs.
        h_s, h_r, h_g = [], [], []
        t = 0
        done = False
        s = to_list(env.reset())
        while (not done) or (t < t_max):
            a = p.predict(de_list(s))
            s_next, r, done, info = env.step(to_list(a))
            # `g` generalized gamma which makes bellman's equaltion applies to
            # hetergenous time delay.
            g = gamma ** (info['time'] if 'time' in info.keys() else 1)
            h_s.append(s)
            h_r.append(r)
            h_g.append(g)
            s = to_list(s_next)
            t += 1
        self.T += t

        # summing gradients is vectorized, It looks not like what's in original
        # paper, but it produces the same result.
        h_g.append(1.0)
        ss = de_list([np.concatenate(z, axis=0)
                      for z in zip(*h_s)])
        rr = np.array(h_r)
        gg = np.array(h_g)
        RR = np.cumsum(rr * gg[1:])
        vv = v.predict(ss).flatten()
        if not done:
            RR += vv[-1:] ** np.cumprod(gg[:-1])

        # In order to broadcast to actions' shape for training,
        # expanding dims is needed.
        diff_RR = RR - vv
        diff_RRs = []
        for sh in p.output_shape:
            tmp = diff_RR
            while tmp.ndim < len(sh):
                tmp = np.expand_dims(tmp, axis=-1)
            diff_RRs.append(tmp)

        # parallel training with lock
        def fff(nn, target):
            nn.fit(ss, target, **kargs)
        self.pool.map_async(
            fff, [(p, diff_RRs),
                  (v, RR)])

        # update_weights with lock
        def ggg(nn, new_nn, nn_w0):
            nn_w = nn.get_weights()
            new_nn_w = new_nn.get_weights()
            assert not (nn_w0[0] is new_nn_w[0])
            nn.set_weights([
                w + (w1 - w0) for w, w0, w1 in
                zip(nn_w, nn_w0, new_nn_w)])
        with self.lock:
            self.pool.map_async(
                ggg, [(self.p, p, p_w0),
                      (self.v, v, v_w0)])

    def train_env(self, T_max, env, gamma=0.9, t_max=np.inf, **kargs):
        """
        Training A3C

        T_max: max num of episodes.
        env: similiar to openai gym but only `env.step` method has
            to be implemeted. The observations and actions of `env.step` must
            be compitable with keras inputs and outputs.

            Notice: actions are preprocessed into a list of nd-array(s) before
                `env.step` applying for convenient.
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
