# Copyright 2018, (Rolinek, Martius)
# MIT License Agreement
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Implementation of L4 stepsize adaptation scheme proposed in (Rolinek, Martius, 2018)"""

import tensorflow as tf


def flatten_list_of_tensors(list_of_tensors):
    return tf.concat([tf.reshape(tensor, [-1]) for tensor in list_of_tensors], axis=0)


def flatten_and_inner_product(list_of_tensors1, list_of_tensors2):
    return tf.tensordot(flatten_list_of_tensors(list_of_tensors1), flatten_list_of_tensors(list_of_tensors2), 1)


def time_factor(time_step):
    """ Routine used for bias correction in exponential moving averages, as in (Kingma, Ba, 2015) """
    global_step = 1 + tf.train.get_or_create_global_step()
    decay = 1.0 - 1.0 / time_step
    return 1.0 - tf.exp((tf.cast(global_step, tf.float32)) * tf.log(decay))


class AdamTransform(object):
    """
    Class implementing Adam (Kingma, Ba 2015) transform of the gradient.
    """
    def __init__(self, time_scale_grad=10.0, time_scale_var=1000.0, epsilon=1e-4):
        self.time_scale_grad = time_scale_grad
        self.time_scale_var = time_scale_var
        self.epsilon = epsilon

        self.EMAgrad = tf.train.ExponentialMovingAverage(decay=1.0 - 1.0 / self.time_scale_grad)
        self.EMAvar = tf.train.ExponentialMovingAverage(decay=1.0 - 1.0 / self.time_scale_var)

    def __call__(self, grads):
        shadow_op_gr = self.EMAgrad.apply(grads)
        vars = [tf.square(grad) for grad in grads]
        shadow_op_var = self.EMAvar.apply(vars)

        with tf.control_dependencies([shadow_op_gr, shadow_op_var]):
            correction_term_1 = time_factor(self.time_scale_grad)
            avg_grads = [self.EMAgrad.average(grad) / correction_term_1 for grad in grads]

            correction_term_2 = time_factor(self.time_scale_var)
            avg_vars = [self.EMAvar.average(var) / correction_term_2 for var in vars]
            return [(grad / (tf.sqrt(var) + self.epsilon)) for grad, var in zip(avg_grads, avg_vars)]


class MomentumTransform(object):
    """
    Class implementing momentum transform of the gradient (here in the form of exponential moving average)
    """
    def __init__(self, time_momentum=10.0):
        self.time_momentum = time_momentum
        self.EMAgrad = tf.train.ExponentialMovingAverage(decay=1.0-1.0/self.time_momentum)
    
    def __call__(self, grads):
        shadow_op_gr = self.EMAgrad.apply(grads)
        with tf.control_dependencies([shadow_op_gr]):
            correction_term = time_factor(self.time_momentum)
            new_grads = [self.EMAgrad.average(grad) / correction_term for grad in grads]
            return [tf.identity(grad) for grad in new_grads]


string_to_transform = {'momentum': MomentumTransform,
                       'adam': AdamTransform}


class L4General(tf.train.GradientDescentOptimizer):
    """
    Class implementing the general L4 stepsize adaptation scheme as a TensorFlow optimizer. The method for applying
    gradients and minimizing a variable are implemented. Note that apply_gradients expects loss as an input parameter.
    """
    def __init__(self, fraction=0.15, minloss_factor=0.9, init_factor=0.75,
                 minloss_forget_time=1000.0, epsilon=1e-12,
                 gradient_estimator='momentum', gradient_params=None,
                 direction_estimator='adam', direction_params=None):
        """
        :param fraction: [alpha], fraction of 'optimal stepsize'
        :param minloss_factor: [gamma], fraction of min seen loss that is considered achievable
        :param init_factor: [gamma_0], fraction of initial loss used to initialize L_min
        :param minloss_forget_time:  [Tau], timescale for forgetting minimum seen loss
        :param epsilon: [epsilon], for numerical stability in the division
        :param gradient_estimator: [g], a gradient method to be used for gradient estimation
        :param gradient_params: dictionary of parameters to pass to gradient_estimator
        :param direction_estimator: [v], a gradient method used for update direction
        :param direction_params: dictionary of parameters to pass to direction_estimator
        """
        tf.train.GradientDescentOptimizer.__init__(self, 1.0)
        with tf.variable_scope('L4Optimizer', reuse=tf.AUTO_REUSE):
            self.min_loss = tf.get_variable(name='min_loss', shape=(),
                                            initializer=tf.constant_initializer(0.0), trainable=False)
        self.fraction = fraction
        self.minloss_factor = minloss_factor
        self.minloss_increase_rate = 1.0 + 1.0 / minloss_forget_time
        self.epsilon = epsilon
        self.init_factor = init_factor

        if not direction_params:
            direction_params = {}
        if not gradient_params:
            gradient_params = {}

        self.grad_direction = string_to_transform[direction_estimator](**direction_params)
        self.deriv_estimate = string_to_transform[gradient_estimator](**gradient_params)
 
    def apply_gradients(self, grads_and_vars, loss, global_step=None, name=None):
        if not global_step:
            global_step = tf.train.get_or_create_global_step()
        grads, vars = zip(*grads_and_vars)

        ml_newval = tf.cond(tf.equal(global_step, 0), lambda: self.init_factor*loss,
                                                      lambda: tf.minimum(self.min_loss, loss))
        ml_update = self.min_loss.assign(ml_newval)

        with tf.control_dependencies([ml_update]):
            directions = self.grad_direction(grads)
            derivatives = self.deriv_estimate(grads)

            min_loss_to_use = self.minloss_factor * self.min_loss
            l_rate = self.fraction*(loss - min_loss_to_use) / (flatten_and_inner_product(directions, derivatives)+self.epsilon)
            new_grads = [direction*l_rate for direction in directions]
            tf.summary.scalar('effective_learning_rate', l_rate)
            tf.summary.scalar('min_loss_estimate', self.min_loss)
            ml_update2 = self.min_loss.assign(self.minloss_increase_rate * self.min_loss)

            with tf.control_dependencies([ml_update2]):
                return tf.train.GradientDescentOptimizer.apply_gradients(self, zip(new_grads, vars), global_step, name)

    def minimize(self, loss, global_step=None, var_list=None, name=None):
        if not var_list:
            var_list = tf.trainable_variables()

        grads_and_vars = self.compute_gradients(loss, var_list)
        return self.apply_gradients(grads_and_vars, loss, global_step, name)


class L4Adam(L4General):
    """
    Specialization of the L4 stepsize adaptation with Adam used for gradient updates and Mom for gradient estimation.
    """
    def __init__(self, fraction=0.15, minloss_factor=0.9, init_factor=0.75, minloss_forget_time=1000.0,
                 epsilon=1e-12, adam_params=None):
        L4General.__init__(self, fraction, minloss_factor, init_factor, minloss_forget_time,
                           epsilon, gradient_estimator='momentum', direction_estimator='adam',
                           direction_params=adam_params)


class L4Mom(L4General):
    """
    Specialization of the L4 stepsize adaptation with Mom used for both gradient estimation and an update direction.
    """
    def __init__(self, fraction=0.15, minloss_factor=0.9, init_factor=0.75, minloss_forget_time=1000.0,
                 epsilon=1e-12, mom_params=None):
        L4General.__init__(self, fraction, minloss_factor, init_factor, minloss_forget_time,
                           epsilon, gradient_estimator='momentum', direction_estimator='momentum',
                           direction_params=mom_params)