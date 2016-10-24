import keras.backend as K
import numpy as np
from keras.models import Model
import keras.optimizers as opts

from multiprocessing import Pool
import threading as z


def to_list(a):
    return a if type(a) is list else [a]


def de_list(a):
    return a[0] if len(a) == 1 else a


class A3C:
    def __init__(self, p, v):
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
        self.t = 0
        self.pool = Pool()

    def compile(self, optimizer_p, optimizer_v):
        opt_p = opts.get(optimizer_p)
        opt_v = opts.get(optimizer_v)

        def loss_p(diff_r_v, a):
            return K.mean(K.log(a) * diff_r_v, -1)

        self.p.compile(optimizer=opt_p, loss=loss_p)
        self.v.compile(optimizer=opt_v, loss='mse')

    def thread_step(self, env, gamma=0.9, t_max=np.inf, **kargs):
        p = Model.from_config(self.p.get_config())
        v = Model.from_config(self.v.get_config())
        p.set_weights(self.p.get_weights())
        v.set_weights(self.v.get_weights())

        h_s, h_R = [], []
        t = 0
        R = np.array([0])
        done = False
        s = env.reset()
        while (not done) or (t < t_max):
            a = p.predict(s)
            s_next, r, done, info = env.step(a)
            R = r + gamma * R
            h_s.append(s)
            h_R.append(R)
            s = s_next
            t += 1
        self.t += t

        ss = [np.concatenate(h, axis=0) for h in h_s]
        RR = np.concatenate(h_R, axis=0)

        if done:
            RR += v.predict(s) ** np.cumsum(np.ones(RR.shape))

        diff_RR = RR - v.predict(ss)
        diff_RRs = []
        for sh in p.output_shapes:
            tmp = diff_RR
            for _ in range(len(sh)-1):
                tmp = np.expand_dims(tmp, axis=-1)
            diff_RRs.append(tmp)

        self.pool.pmap(
            lambda z, b: z.fit(ss, b, **kargs),
            [(p, diff_RRs), (v, RR)])

        with self.p.lock, self.v.lock:
            self.p.set_weights(p.get_weights())
            self.v.set_weights(v.get_weights())
