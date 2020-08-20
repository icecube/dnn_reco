import tensorflow as tf

from dnn_reco import misc


class MultiLearningRateScheduler(tf.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that combines multiple schedulers
    """

    def __init__(self, boundaries, scheduler_settings, name=None):
        """MultiLearningRateScheduler

        The function returns a 1-arg callable to compute the multi learning
        rate schedule when passed the current optimizer step.


        Parameters
        ----------
        boundaries
            A list of `Tensor`s or `int`s or `float`s with strictly
            increasing entries, and with all elements having the same type as
            the optimizer step.
        scheduler_settings : list of dict
            A list of scheduler settings that specify the learning rate
            schedules to use for the intervals defined by `boundaries`.
            It should have one more element than `boundaries`, and all
            schedulers should return the same type.
            Each scheduler_setting dict should contain the following:
                'full_class_string': str
                    The full class string of the scheduler class
                'settings': dict
                    A dictionary of arguments that are passed on to
                    the scheduler class
        name
            A string. Optional name of the operation. Defaults to
            'MultiLearningRateScheduler'.
        """

        super(MultiLearningRateScheduler, self).__init__()

        if len(boundaries) != len(scheduler_settings) - 1:
            raise ValueError(
              "The length of boundaries should be 1 less than the length "
              "of scheduler_settings")

        # create schedulers
        schedulers = []
        for settings in scheduler_settings:
            scheduler_class = misc.load_class(settings['full_class_string'])
            scheduler = scheduler_class(**settings['settings'])
            schedulers.append(scheduler)

        if name is None:
            name = 'MultiLearningRateScheduler'

        self.boundaries = tf.convert_to_tensor(boundaries)
        self.scheduler_settings = scheduler_settings
        self.schedulers = schedulers
        self.name = name

    @tf.function
    def __call__(self, step):

        step = tf.convert_to_tensor(step)
        boundaries = tf.cast(self.boundaries, step.dtype)

        # create a list of (boolean, callable) pairs
        pred_fn_pairs = []

        pred_fn_pairs.append((
            step <= boundaries[0],
            lambda: self.schedulers[0](step),
        ))
        pred_fn_pairs.append((
            step > boundaries[-1],
            lambda: self.schedulers[-1](step - boundaries[-1]),
        ))
        for index in range(len(self.schedulers) - 2):
            low = boundaries[index]
            high = boundaries[index + 1]
            scheduler = self.schedulers[index + 1]

            pred_fn_pairs.append((
                step > low and step <= high,
                lambda: scheduler(step - low),
            ))

        return tf.case(pred_fn_pairs, exclusive=True)

    def get_config(self):
        return {
            "boundaries": self.boundaries,
            "scheduler_settings": self.scheduler_settings,
            "name": self.name
        }
