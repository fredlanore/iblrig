from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import iblrig.misc
from iblrig.base_choice_world import ActiveChoiceWorldSession

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)


class Session(ActiveChoiceWorldSession):
    """
    Advanced Choice World is the ChoiceWorld task using fixed 50/50 probability for the side
    and contrasts defined in the parameters.
    It differs from TrainingChoiceWorld in that it does not implement adaptive contrasts or debiasing,
    and it differs from BiasedChoiceWorld in that it does not implement biased blocks.
    """

    protocol_name = '_iblrig_tasks_advancedChoiceWorld'

    def __init__(
        self,
        *args,
        contrast_set: list[float] = DEFAULTS['CONTRAST_SET'],
        probability_set: float | list[float] = DEFAULTS['PROBABILITY_SET'],
        reward_set_ul: float | list[float] = DEFAULTS['REWARD_SET_UL'],
        position_set: list[float] = DEFAULTS['POSITION_SET'],
        stim_gain: float = DEFAULTS['STIM_GAIN'],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_params['CONTRAST_SET'] = contrast_set
        self.task_params['PROBABILITY_SET'] = probability_set
        self.task_params['REWARD_SET_UL'] = reward_set_ul
        self.task_params['POSITION_SET'] = position_set
        self.task_params['STIM_GAIN'] = stim_gain
        nc = len(self.task_params['CONTRAST_SET'])
        self.df_contrasts = pd.DataFrame(columns=['contrast', 'probability', 'reward_amount_ul', 'position'])
        self.df_contrasts['contrast'] = self.task_params['CONTRAST_SET']

    def draw_next_trial_info(self, **kwargs):
        nc = len(self.task_params['CONTRAST_SET'])
        ic = np.random.choice(np.arange(nc), p=self.task_params['PROBABILITY_SET'])
        self.task_params['PROBABILITY_SET']
        pleft, contrast, position= (0.5, 0, -35)
        super().draw_next_trial_info(pleft=pleft, contrast=contrast, position=position)
    @property
    def reward_amount(self):
        return self.task_params.REWARD_AMOUNTS_UL[0]

    @staticmethod
    def extra_parser():
        """:return: argparse.parser()"""
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--contrast_set',
            option_strings=['--contrast_set'],
            dest='contrast_set',
            default=DEFAULTS['CONTRAST_SET'],
            nargs='+',
            type=list[float],
            help='set of contrasts to present',
        )
        parser.add_argument(
            '--probability_set',
            option_strings=['--probability_set'],
            dest='probability_set',
            default=DEFAULTS['PROBABILITY_SET'],
            nargs='+',
            type=float | list[float],
            help='probabilities of each contrast in contrast_set. If scalar all contrasts are equiprobable',
        )
        parser.add_argument(
            '--reward_set_ul',
            option_strings=['--reward_set_ul'],
            dest='reward_set_ul',
            default=DEFAULTS['REWARD_SET_UL'],
            type=float | list[float],
            help=f'reward amount (default: {DEFAULTS["REWARD_SET_UL"]}μl), can be a vector of n contrasts or a scalar',
        )
        parser.add_argument(
            '--position_set',
            option_strings=['--position_set'],
            dest='probability_set',
            default=DEFAULTS['PROBABILITY_SET'],
            nargs='+',
            type=list[float],
            help='Position for each contrast in contrast set.',
        )
        parser.add_argument(
            '--stim_gain',
            option_strings=['--stim_gain'],
            dest='stim_gain',
            default=DEFAULTS['STIM_GAIN'],
            type=float,
            help=f'visual angle/wheel displacement ' f'(deg/mm, default: {DEFAULTS["STIM_GAIN"]})',
        )
        return parser

    def next_trial(self):
        # update counters
        self.trial_num += 1
        # save and send trial info to bonsai
        self.draw_next_trial_info(pleft=self.task_params.PROBABILITY_LEFT)


if __name__ == '__main__':  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()
