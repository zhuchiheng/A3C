import keras.backend as K
import numpy as np
from keras.models import Model

from multiprocessing import Pool
import threading as z


def to_list(a):
    return a if type(a) is list else [a]


def de_list(a):
    return a[0] if len(a) == 1 else a


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

        self.p = p
        self.v = v
        self.T = 0
        self.pool = Pool()

    def compile(self, optimizer):
        """compiles keras models, with keras optimizer"""

        def loss_p(diff_r_v, a):
            return K.mean(K.log(a) * diff_r_v, -1)

        self.p.compile(optimizer=optimizer, loss=loss_p)
        self.v.compile(optimizer=optimizer, loss='mse')

    def thread_step(self, env, gamma=0.9, t_max=np.inf, **kargs):
        # clone models for async training
        p = Model.from_config(self.p.get_config())
        v = Model.from_config(self.v.get_config())
        p.set_weights(self.p.get_weights())
        v.set_weights(self.v.get_weights())

        # loop of states->acts, `env.step` should be compatible with keras'
        # inputs.
        h_s, h_r, h_g = [], [], []
        t = 0
        done = False
        s = env.reset()
        while (not done) or (t < t_max):
            a = p.predict(s)
            s_next, r, done, info = env.step(a)
            # `g` generalized gamma which makes bellman's equaltion applies to
            # hetergenous time delay.
            g = gamma ** (info['time'] if 'time' in info.keys() else 1)
            h_s.append(s)
            h_r.append(r)
            h_g.append(g)
            s = s_next
            t += 1
        self.T += t

        # summing gradients is vectorized, It looks not like what's in original
        # paper, but it produces the same result.
        h_g.append(1)
        ss = zip(*[np.concatenate(z, axis=0) for z in zip(*h_s)])
        rr = np.array(h_r)
        gg = np.array(h_g)
        RR = np.cumsum(rr * gg[1:])
        if not done:
            RR += v.predict(s) ** np.cumprod(gg[:-1])

        # In order to broadcast to actions' shape for training,
        # expanding dims is needed.
        diff_RR = RR - v.predict(ss)
        diff_RRs = []
        for sh in p.output_shapes:
            tmp = diff_RR
            for _ in range(len(sh)-1):
                tmp = np.expand_dims(tmp, axis=-1)
            diff_RRs.append(tmp)

        # parallel training with lock
        with self.p.lock, self.v.lock:
            self.pool.pmap(
                lambda z, b: z.fit(ss, b, **kargs),
                [(p, diff_RRs), (v, RR)])
