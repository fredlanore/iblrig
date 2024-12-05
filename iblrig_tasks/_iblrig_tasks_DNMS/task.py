"""
 DNMS (Delayed Non-Match-To-Sample) task
"""
from pathlib import Path

import logging
import pydantic
import pybpodapi.protocol
import numpy as np
import pandas as pd
import yaml
from pydantic import NonNegativeFloat

import iblrig.misc
from iblrig.base_tasks import BaseSession #what world to import? Or rather create a new one? Or simply implement the new class called DNMS session inside of the base_choice_world?
from iblrig.hardware import SOFTCODE
from pybpodapi.protocol import StateMachine
# from iblrig.base_choice_world import NTRIALS_INIT, ActiveChoiceWorldSession

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)

class DNMSSession(BaseSession):
     """
    DNMS Choice World is the ChoiceWorld task targeting working memory processes. It happens in several epochs:
    1. Precue presentation: a visual stimulus is shown on the screen
    2. Delay: a gray screen is shown
    3. Cue presentation: 2 visual stimuli are shown, one of them is identical to the precue and another is the mirror image of the precue
    4. Closed loop: the mouse has to choose between the two stimuli. Possible outcomes are correct choice, error, or no-go.
    5. Feedback: the mouse gets a reward if it chooses the correct stimulus, i.e. the one that mismatches the precue
    6. ITI: the mouse waits for the next trial
    It differs from BaseChoiceWorld in that it implements the additional epochs and shows 2 visual stimuli on the screen.
    """
    protocol_name = '_iblrig_tasks_DelayedNonMatchToSample'
    #TrialDataModel = DNMSChoiceTrialData

    def __init__(
        self,
        *args,
        contrast_set: float = DEFAULTS['CONTRAST_SET'],
        #probability_set: list[float] = DEFAULTS['PROBABILITY_SET'],
        #stim_reverse: float = DEFAULTS['STIM_REVERSE'],
        reward_set_ul: float = DEFAULTS['REWARD_SET_UL'],
        stim_gain: float = DEFAULTS['STIM_GAIN'],
        precue_position_set: list[float] = DEFAULTS['PRECUE_POSITION_SET'],
        precue_presentation: float = DEFAULTS['PRECUE_PRESENTATION'],
        delay_duration: list[float] =DEFAULTS['DELAY_DURATION_SET'],
        correct_cue_position_set: list[float] = DEFAULTS['CORRECT_CUE_POSITION_SET'],
        cue_presentation: float = DEFAULTS['CUE_PRESENTATION'],
        timeout: float = DEFAULTS['TIMEOUT'],
        iti_duration: float = DEFAULTS['ITI_DURATION'],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        nc = len(delay_duration)
        assert len(precue_position_set) in [nc, 1], 'precue_position_set must be a scalar or have the same length as delay_duration'
        assert len(correct_cue_position_set) == nc, 'correct_cue_position_set must have the same length as delay_duration'
        assert len(reward_set_ul) in [nc, 1], 'reward_set_ul must be a scalar or have the same length as delay_duration'
        self.task_params['CONTRAST_SET'] = contrast_set
        #self.task_params['PROBABILITY_SET'] = probability_set
        #self.task_params['STIM_REVERSE'] = stim_reverse
        self.task_params['STIM_GAIN'] = stim_gain
        self.task_params['REWARD_SET_UL'] = reward_set_ul
        self.task_params['PRECUE_POSITION_SET'] = precue_position_set
        self.task_params['PRECUE_PRESENTATION'] = precue_presentation
        self.task_params['DELAY_DURATION'] = delay_duration
        self.task_params['CORRECT_CUE_POSITION_SET'] = correct_cue_position_set
        self.task_params['CUE_PRESENTATION'] = cue_presentation
        self.task_params['TIMEOUT'] = timeout
        self.task_params['ITI_DURATION'] = iti_duration
        
        #should I put all the parameters in the dataframe..?
        self.df_contingencies = pd.DataFrame(columns=['delay', 'correct_cue_position', 'reward_amount_ul'])
        self.df_contingencies['delay'] = delay_duration
        self.df_contingencies['correct_cue_position'] = correct_cue_position_set
        self.df_contingencies['reward_amount_ul'] = reward_set_ul if len(reward_set_ul) == nc else reward_set_ul[0]
        #self.df_contingencies['contrast'] = contrast_set if len(contrast_set) == nc else contrast_set[0]


    def draw_next_trial_info(self, **kwargs):
        nc = self.df_contingencies.shape[0]
        ic = np.random.choice(np.arange(nc), p=self.df_contingencies['probability'])
        # now calling the super class with the proper parameters
        super().draw_next_trial_info(
            delay = self.df_contingencies.at[ic, 'delay'],
            correct_cue_position=self.df_contingencies.at[ic, 'correct_cue_position'],
            reward_amount=self.df_contingencies.at[ic, 'reward_amount_ul'],
        )
    
    @property
    def reward_amount(self):
        return self.task_params.REWARD_AMOUNTS_UL[0] 
    
    @property
    def correct_cue_position(self):
        return self.task_params.CORRECT_CUE_POSITION_SET[0] #unsure about the index, should it be deleted?
    
    @property
    def delay_duration(self):
        return self.task_params.DELAY_DURATION_SET[0] #unsure about the index, should it be deleted?
    
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
            type=float,
            help='Set of contrasts to present',
        )
        
        parser.add_argument(
            '--reward_set_ul',
            option_strings=['--reward_set_ul'],
            dest='reward_set_ul',
            default=DEFAULTS['REWARD_SET_UL'],
            nargs='+',
            type=float,
            help='Reward for contrast in contrast set.',
        )
        parser.add_argument(
            '--correct_cue_position_set',
            option_strings=['--correct_cue_position_set'],
            dest='correct_cue_position_set',
            default=DEFAULTS['CORRECT_CUE_POSITION_SET'],
            nargs='+',
            type=float,
            help='Position for each correct cue in correct_cue set.',
        )
        parser.add_argument(
            '--stim_gain',
            option_strings=['--stim_gain'],
            dest='stim_gain',
            default=DEFAULTS['STIM_GAIN'],
            type=float,
            help=f'Visual angle/wheel displacement ' f'(deg/mm, default: {DEFAULTS["STIM_GAIN"]})',
        )

        return parser

    def next_trial(self):
        # update counters
        self.trial_num += 1
        # save and send trial info to bonsai
        self.draw_next_trial_info(pleft=self.task_params.PROBABILITY_LEFT)


    def get_state_machine_trial(self, i):
        sma = StateMachine(self.bpod)
        log = logging.getLogger(__name__)

        #FIRST TRIAL
        if i == 0:  # First trial exception start camera
            session_delay_start = self.task_params.get('SESSION_DELAY_START', 0)
            log.info('First trial initializing, will move to next trial only if:')
            log.info('1. camera is detected')
            log.info(f'2. {session_delay_start} sec have elapsed')
            sma.add_state(
                state_name='trial_start',
                state_timer=0,
                state_change_conditions={'Port1In': 'delay_initiation'},
                output_actions=[('SoftCode', SOFTCODE.TRIGGER_CAMERA), ('BNC1', 255)],
            )  # start camera
            sma.add_state(
                state_name='delay_initiation',
                state_timer=session_delay_start,
                output_actions=[],
                state_change_conditions={'Tup': 'reset_rotary_encoder'},
            )
        else:
            sma.add_state(
                state_name='trial_start',
                state_timer=0,  # ~100µs hardware irreducible delay
                state_change_conditions={'Tup': 'reset_rotary_encoder'},
                output_actions=[self.bpod.actions.stop_sound, ('BNC1', 255)],
            )  # stop all sounds

        sma.add_state(
            state_name='reset_rotary_encoder',
            state_timer=0,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={'Tup': 'quiescent_period'},
        )

        # Quiescent Period. If the wheel is moved past one of the thresholds: Reset the rotary encoder and start over.
        # Continue with the stimulation once the quiescent period has passed without triggering movement thresholds.
        sma.add_state(  # '>back' | '>reset_timer'
            state_name='quiescent_period',
            state_timer=self.quiescent_period,
            output_actions=[],
            state_change_conditions={
                'Tup': 'precue_on',
                self.movement_left: 'reset_rotary_encoder',
                self.movement_right: 'reset_rotary_encoder',
            },
        )

        # PRECUE EPOCH
        # Show the visual stimulus. This is achieved by sending a time-stamped byte-message to Bonsai via the Rotary
        # Encoder Module's ongoing USB-stream. Move to the next state once the Frame2TTL has been triggered, i.e.,
        # when the stimulus has been rendered on screen. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name='precue_on',
            state_timer=self.task_params.PRECUE_PRESENTATION, #should I define state_timer like this or by calling precue_presentation variable?
            output_actions=[self.bpod.actions.bonsai_show_stim],
            state_change_conditions={'Tup': 'interactive_delay', 'BNC1High': 'interactive_delay', 'BNC1Low': 'interactive_delay'},
        )

        # Defined delay between visual and auditory cues (could the presentation of auditive and visual cues be merged into one sma?)
        sma.add_state(
            state_name='interactive_delay',
            state_timer= 0.05
            output_actions=[],
            state_change_conditions={'Tup': 'play_tone'},
        )

        # Play tone. Move to next state if sound is detected. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name='play_tone',
            state_timer=0.1,
            output_actions=[self.bpod.actions.play_tone],
            state_change_conditions={'Tup': 'reset2_rotary_encoder', 'BNC2High': 'reset2_rotary_encoder'},
        )

        sma.add_state(
            state_name='precue_off',
            state_timer=0.1
            output_actions=[self.bpod.actions.bonsai_hide_stim],
            state_change_conditions={'Tup': 'reset2_rotary_encoder', 'BNC1High': 'reset2_rotary_encoder', 'BNC1Low': 'reset2_rotary_encoder'},
        )

        # Reset rotary encoder (see above). Move on after brief delay (to avoid a race conditions in the bonsai flow).
        sma.add_state(
            state_name='reset2_rotary_encoder',
            state_timer=0.05,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={'Tup': 'delay_on'},
        )

        # DELAY EPOCH
        sma.add_state(
            state_name='delay_on',
            state_timer=self.task_params.DELAY_DURATION_SET, #same question as previously for the precue-on sma
            output_actions=[('delay duration', self.task_params.DELAY_DURATION_SET)], #is it necessary to call upon bonsai to show the gray background that we see during the cue presentation?
            state_change_conditions={'Tup': 'cue_on'},
        )

        # CUE EPOCH
        sma.add_state(
            state_name='cue_on',
            state_timer=self.task_params.CUE_PRESENTATION, #same question as previously for the precue-on sma
            output_actions=[self.bpod.actions.bonsai_show_stim, ('cue presentation', self.task_params.CUE_PRESENTATION)],
            state_change_conditions={'Tup': 'reset2_rotary_encoder', 'BNC1High': 'reset2_rotary_encoder', 'BNC1Low': 'reset2_rotary_encoder'},
        )

        sma.add_state(
            state_name='reset2_rotary_encoder',
            state_timer=0.05,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={'Tup': 'closed_loop'},
        )

        #CHOICE EPOCH
        # Start the closed loop state in which the animal controls the position of the visual stimulus by means of the
        # rotary encoder. The three possible outcomes are:
        # 1) wheel has NOT been moved past a threshold: continue with no-go condition
        # 2) wheel has been moved in WRONG direction: continue with error condition
        # 3) wheel has been moved in CORRECT direction: continue with reward condition
        sma.add_state(
            state_name='closed_loop',
            state_timer=self.task_params.TIMEOUT,
            output_actions=[self.bpod.actions.bonsai_closed_loop],
            state_change_conditions={'Tup': 'no-go', self.event_reward:'correct_choice', self.event_error:'false_choice'},
        )

        # No-go: hide the visual stimulus and play white noise. Go to exit_state after FEEDBACK_NOGO_DELAY_SECS.
        sma.add_state(
            state_name='no_go',
            state_timer=self.feedback_nogo_delay,
            output_actions=[self.bpod.actions.bonsai_hide_stim, self.bpod.actions.play_noise],
            state_change_conditions={'Tup': 'exit_state'},
        )

        # Error: Freeze the stimulus and play white noise.
        # Continue to hide_stim/exit_state once FEEDBACK_ERROR_DELAY_SECS have passed.
        sma.add_state(
            state_name='false_choice',
            state_timer=0,
            output_actions=[self.bpod.actions.bonsai_freeze_stim],
            state_change_conditions={'Tup': 'error'},
        )
        sma.add_state(
            state_name='error',
            state_timer=self.feedback_error_delay,
            output_actions=[self.bpod.actions.play_noise],
            state_change_conditions={'Tup': 'hide_stim'},
        )

        # Reward: open the valve for a defined duration (and set BNC1 to high), freeze stimulus in center of screen.
        # Continue to hide_stim/exit_state once FEEDBACK_CORRECT_DELAY_SECS have passed.
        sma.add_state(
            state_name='correct_choice',
            state_timer=0,
            output_actions=[self.bpod.actions.bonsai_show_center],
            state_change_conditions={'Tup': 'reward'},
        )
        sma.add_state(
            state_name='reward',
            state_timer=self.reward_time,
            output_actions=[('Valve1', 255), ('BNC1', 255)],
            state_change_conditions={'Tup': 'correct'},
        )
        sma.add_state(
            state_name='correct',
            state_timer=self.feedback_correct_delay - self.reward_time,
            output_actions=[],
            state_change_conditions={'Tup': 'hide_stim'},
        )

        # Hide the visual stimulus. This is achieved by sending a time-stamped byte-message to Bonsai via the Rotary
        # Encoder Module's ongoing USB-stream. Move to the next state once the Frame2TTL has been triggered, i.e.,
        # when the stimulus has been rendered on screen. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name='hide_stim',
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_hide_stim],
            state_change_conditions={'Tup': 'exit_state', 'BNC1High': 'exit_state', 'BNC1Low': 'exit_state'},
        )

        # Wait for ITI_DELAY_SECS before ending the trial. Raise BNC1 to mark this event.
        sma.add_state(
            state_name='exit_state',
            state_timer=self.task_params.ITI_DURATION,
            output_actions=[('BNC1', 255), ('ITI duration', self.task_params.ITI_DURATION)],
            state_change_conditions={'Tup': 'exit'},
        )

        return sma

#launches the task
if __name__ == '__main__':  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()
