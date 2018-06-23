from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from ray.tune import TuneError
from ray.tune.experiment import Experiment
from ray.tune.log_sync import wait_for_log_sync
from ray.tune.trial import Trial, DEBUG_PRINT_INTERVAL
from ray.tune.trial_runner import TrialRunner
from ray.tune.trial_scheduler import FIFOScheduler
from ray.tune.variant_generator import generate_trials
from ray.tune.web_server import TuneServer


class TrialRunnerV2(TrialRunner):
    def has_resources(self, resources):
        """Returns whether this runner has at least the specified resources."""

        cpu_avail = self._avail_resources.cpu - self._committed_resources.cpu
        #gpu_avail = self._avail_resources.gpu - self._committed_resources.gpu
        return ( resources.cpu_total() <= cpu_avail)



def run_experiments(experiments, scheduler=None, with_server=False,
                    server_port=TuneServer.DEFAULT_PORT, verbose=True):
    """Tunes experiments.

    Args:
        experiments (Experiment | list | dict): Experiments to run.
        scheduler (TrialScheduler): Scheduler for executing
            the experiment. Choose among FIFO (default), MedianStopping,
            AsyncHyperBand, or HyperBand.
        with_server (bool): Starts a background Tune server. Needed for
            using the Client API.
        server_port (int): Port number for launching TuneServer.
        verbose (bool): How much output should be printed for each trial.
    """

    # Make sure rllib agents are registered

    if scheduler is None:
        scheduler = FIFOScheduler()

    runner = TrialRunnerV2(
        scheduler, launch_web_server=with_server, server_port=server_port)

    if type(experiments) is dict:
        for name, spec in experiments.items():
            for trial in generate_trials(spec, name):
                trial.set_verbose(verbose)
                runner.add_trial(trial)
    elif (type(experiments) is list and
          all(isinstance(exp, Experiment) for exp in experiments)):
        for experiment in experiments:
            for trial in experiment.trials():
                trial.set_verbose(verbose)
                runner.add_trial(trial)
    elif isinstance(experiments, Experiment):
        for trial in experiments.trials():
            trial.set_verbose(verbose)
            runner.add_trial(trial)

    print(runner.debug_string(max_debug=99999))

    last_debug = 0
    while not runner.is_finished():
        runner.step()
        if time.time() - last_debug > DEBUG_PRINT_INTERVAL:
            print(runner.debug_string())
            last_debug = time.time()

    print(runner.debug_string(max_debug=99999))

    for trial in runner.get_trials():
        # TODO(rliaw): What about errored?
        if trial.status != Trial.TERMINATED:
            raise TuneError("Trial did not complete", trial)

    wait_for_log_sync()
    return runner.get_trials()
