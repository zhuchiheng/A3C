import keras.backend as K
import numpy as np
from keras.models import Model


class A3C:
    def __init__(self, p, v):
        self.p = p
        self.v = v
        self.t = 0

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
            K.function(
                p.inputs + [x],
                [K.gradients(K.sum(K.log(aa) * x), w)
                 for aa in p.outputs]
            ) for w in p.trainable_weights]
        dd_v = [
            K.function(
                v.inputs + [x],
                [K.gradients((x - v.outputs[0])**2, w)]
            ) for w in p.trainable_weights]
        for s, a, r in h:
            R = r + gm * R
            d_th_p = [w + sum(dd(R - v.predict(s)))
                      for dd, w in zip(dd_p, d_th_p)]
            d_th_v = [w + dd_v(R)[0] for w in d_th_v]

            self.async_update(d_th_p, d_th_v)

    def async_update(d_th_p, d_th_v):
        raise NotImplementedError
