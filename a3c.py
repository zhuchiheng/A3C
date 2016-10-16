import keras.backend as K
import numpy as np
from keras.models import Model
import keras.optimizers as opts
import threading as z


def to_list(a):
    return a if type(a) is list else [a]


def de_list(a):
    return a[0] if len(a) == 1 else a


class A3C:
    def __init__(self, p, v):
        if (not hasattr(p, "lock")):
            p.lock = z.Lock()
        if (not hasattr(v, "lock")):
            v.lock = z.Lock()
        self.p = p
        self.v = v
        self.t = 0
        self.p_fake_x = de_list([np.zeros(s)
                                 for s in to_list(self.p.input_shape)])
        self.v_fake_x = de_list([np.zeros(s)
                                 for s in to_list(self.v.input_shape)])
        self.p_fake_y = de_list([np.zeros(s)
                                 for s in to_list(self.p.output_shape)])
        self.v_fake_y = de_list([np.zeros(s)
                                 for s in to_list(self.v.output_shape)])
        self.lock = z.Lock()

    def compile(self, optimizer_p, optimizer_v):
        opt_p = opts.get(optimizer_p)
        opt_v = opts.get(optimizer_v)

        def get_gradients(s, _1, _2):
            return s.grads

        def loss(_1, _2):
            return 0

        opt_p.get_gradients = get_gradients
        opt_v.get_gradients = get_gradients
        self.p.compile(optimizer=opt_p, loss=loss)
        self.v.compile(optimizer=opt_v, loss=loss)

    def thread_step(self, env, gm=0.9, t_max=np.inf):
        d_th_p = [np.zeros(w.shape)
                  for w in self.p.trainable_weights]
        d_th_v = [np.zeros(w.shape)
                  for w in self.v.trainable_weights]
        p = Model.from_config(self.p.get_config())
        v = Model.from_config(self.v.get_config())
        p.set_weights(self.p.get_weights())
        v.set_weights(self.v.get_weights())

        h = []

        t = 0
        done = False
        s = env.reset()
        while (not done) or (t < t_max):
            a = p.predict(s)
            s, r, done, info = env.step(a)
            h.append((s, a, r))
            t += 1
        self.t += t

        R = 0 if done else v.predict(s)

        x = K.placeholder(shape=(1,))
        dd_p = [
            [K.gradients(K.sum(K.log(aa)), w)
             for aa in p.outputs]
            for w in p.trainable_weights]
        dd_v = [
            K.gradients((x - v.outputs[0])**2, w)
            for w in p.trainable_weights]
        for s, a, r in h:
            R = r + gm * R
            RR = R - v.predict(s)
            d_th_p = [w + sum(dd) * RR
                      for dd, w in zip(dd_p, d_th_p)]
            d_th_v = [w + dd_v * R for w in d_th_v]

        self.p.optimizer.grads = d_th_p
        self.v.optimizer.grads = d_th_v
        p.train_on_batch(self.p_fake_x, self.p_fake_y)
        v.train_on_batch(self.v_fake_x, self.v_fake_y)

        with self.p.lock, self.v.lock:
            self.p.set_weights(p.get_weights())
            self.v.set_weights(v.get_weights())
