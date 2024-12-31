
"""
 DNMS (Delayed Non-Match-To-Sample) task
"""
from pathlib import Path

import logging
import numpy as np
import pandas as pd
import yaml

import iblrig.misc
import iblrig.base_choice_world
from iblrig.base_choice_world import NTRIALS_INIT, ActiveChoiceWorldSession, ActiveChoiceWorldTrialData
from iblrig.hardware import SOFTCODE
from typing import Annotated, Any
from pybpodapi.protocol import StateMachine
from pydantic import NonNegativeFloat, NonNegativeInt
from annotated_types import Interval, IsNan


# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)

class DNMSChoiceWorldTrialData(ActiveChoiceWorldTrialData):

    # add other variables as well..?
    delay_duration: NonNegativeFloat
    precue_angle: Annotated[float, Interval(ge=-180.0, le=180.0)]
    iti_duration: NonNegativeFloat
    timeout: NonNegativeFloat
    correct_cue_position: float 
    cue_presentation_time: NonNegativeFloat

    response_side: Annotated[int, Interval(ge=-1, le=1)]
    response_time: NonNegativeFloat
    trial_correct: bool
    

   
class DNMSSession(ActiveChoiceWorldSession):
     """
    DNMS Choice World is the ChoiceWorld task engaging working memory processes. It contains several epochs:
    1. Precue presentation: a visual stimulus is shown on the screen
    2. Delay: a gray screen is shown
    3. Cue presentation: 2 visual stimuli are shown, one of them identical to the precue and another the mirror image of the precue
    4. Closed loop: the mouse has to choose between the two stimuli. Possible outcomes are correct choice, error, or no-go.
    5. Feedback: the mouse gets a reward if it chooses the correct stimulus, i.e. the one that mismatches the precue
    6. ITI: the mouse waits for the next trial
    It differs from BaseChoiceWorld in that it implements the additional epochs and shows 2 visual stimuli on the screen.
    """
     protocol_name = '_iblrig_tasks_DelayedNonMatchToSample'
     TrialDataModel = DNMSChoiceWorldTrialData

     def __init__(
        self,
        *args,
        contrast_set: float = DEFAULTS['CONTRAST_SET'],
        precue_angle_set: float = DEFAULTS['PRECUE_ANGLE_SET'], 
        #probability_set: list[float] = DEFAULTS['PROBABILITY_SET'],
        #stim_reverse: float = DEFAULTS['STIM_REVERSE'],
        reward_set_ul: float = DEFAULTS['REWARD_SET_UL'],
        stim_gain: float = DEFAULTS['STIM_GAIN'],
        precue_position_set: list[float] = DEFAULTS['PRECUE_POSITION_SET'],
        precue_presentation_time: float = DEFAULTS['PRECUE_PRESENTATION_TIME'],
        delay_duration: list[float] =DEFAULTS['DELAY_DURATION_SET'],
        correct_cue_position_set: list[float] = DEFAULTS['CORRECT_CUE_POSITION_SET'],
        cue_presentation_time: float = DEFAULTS['CUE_PRESENTATION_TIME'],
        timeout: float = DEFAULTS['TIMEOUT'],
        iti_duration: float = DEFAULTS['ITI_DURATION'],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        nc = len(precue_angle_set)
        assert len(delay_duration) in [nc, 1], 'precue_position_set must be a scalar or have the same length as delay_duration'
        assert len(precue_position_set) in [nc, 1], 'precue_position_set must be a scalar or have the same length as delay_duration'
        assert len(correct_cue_position_set) == nc, 'correct_cue_position_set must have the same length as delay_duration'
        assert len(precue_angle_set) in [nc, 1], 'precue orientation set must be a scalar or have the same length as delay_duration'
        assert len(reward_set_ul) in [nc, 1], 'reward_set_ul must be a scalar or have the same length as delay_duration'
        
        # variables remaining constant from trial to trial should be stored 
        self.task_params['CONTRAST_SET'] = contrast_set
        self.task_params['STIM_GAIN'] = stim_gain
        self.task_params['REWARD_SET_UL'] = reward_set_ul
        self.task_params['PRECUE_POSITION_SET'] = precue_position_set
        self.task_params['PRECUE_PRESENTATION_TIME'] = precue_presentation_time
        self.task_params['CUE_PRESENTATION_TIME'] = cue_presentation_time
        self.task_params['TIMEOUT'] = timeout
        self.task_params['ITI_DURATION'] = iti_duration
        #self.task_params['PROBABILITY_SET'] = probability_set
        #self.task_params['STIM_REVERSE'] = stim_reverse

        #variables that change from trial to trial should be stored into dataframe 
        self.df_contingencies = pd.DataFrame(columns=['precue_angle', 'delay', 'correct_cue_position'])
        self.df_contingencies['precue_angle'] = precue_angle_set if len(precue_angle_set) == nc else precue_angle_set[0]
        self.df_contingencies['delay'] = delay_duration
        self.df_contingencies['correct_cue_position'] = correct_cue_position_set
        #self.df_contingencies['contrast'] = contrast_set if len(contrast_set) == nc else contrast_set[0]


     def draw_next_trial_info(self, pleft=0.5, **kwargs):
        nc = self.df_contingencies.shape[0]
        # now calling the super class with the proper parameters
        super().draw_next_trial_info(
            precue_angle = self.df_contingencies.at[nc, 'precue_angle'],
            delay = self.df_contingencies.at[nc, 'delay'],
            correct_cue_position=self.df_contingencies.at[nc, 'correct_cue_position'],
        )
    
     @property
     def reward_amount(self):
        return self.task_params.REWARD_AMOUNTS_UL[0] 
     
     @property
     def precue_angle(self):
        #return self.task_params.PRECUE_ANGLE_SET 
        return self.df_contingencies['precue_angle']
     
     @property
     def delay_duration(self):
        #return self.task_params.DELAY_DURATION_SET
        return self.df_contingencies['delay']
    
     @property
     def correct_cue_position(self):
        #return self.task_params.CORRECT_CUE_POSITION_SET 
        return self.df_contingencies['correct_cue_position']
    
     
    
     @staticmethod
     def extra_parser():
        """:return: argparse.parser()"""
        parser = super(DNMSSession, DNMSSession).extra_parser()

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
            '--precue_angle_set',
            option_strings=['--precue_angle_set'],
            dest='precue_angle_set',
            default=DEFAULTS['PRECUE_ANGLE_SET'],
            nargs='+',
            type=float,
            help='Angle for each precue.',
        )

        parser.add_argument(
            '--delay_duration',
            option_strings=['--delay_duration'],
            dest='delay_duration',
            default=DEFAULTS['DELAY_DURATION_SET'],
            nargs='+',
            type=float,
            help='Delay for each epoch.',
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
        self.draw_next_trial_info(pleft=0.5)


     def get_state_machine_trial(self, i):
        sma = StateMachine(self.bpod)
        log = logging.getLogger(__name__)

        #FIRST TRIAL
        if i == 0:  # First trial exception start camera
            session_delay_start = self.task_params.get('SESSION_DELAY_START', 0)
            log.info(f"Starting trial {i + 1}")
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

        log.info('Quiescent period OFF')

        # PRECUE EPOCH
        # Show the visual stimulus. This is achieved by sending a time-stamped byte-message to Bonsai via the Rotary
        # Encoder Module's ongoing USB-stream. Move to the next state once the Frame2TTL has been triggered, i.e.,
        # when the stimulus has been rendered on screen. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name='precue_on',
            state_timer=self.task_params.PRECUE_PRESENTATION_TIME,
            output_actions=[self.bpod.actions.bonsai_show_stim],
            state_change_conditions={'Tup': 'interactive_delay', 'BNC1High': 'interactive_delay', 'BNC1Low': 'interactive_delay'},
        )

        log.info('Precue ON for ' + str(self.task_params.PRECUE_PRESENTATION_TIME) + ' seconds')

        # Defined delay between visual and auditory cues (could the presentation of auditive and visual cues be merged into one sma?)
        sma.add_state(
            state_name='interactive_delay',
            state_timer= 0.05,
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
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_hide_stim],
            state_change_conditions={'Tup': 'reset2_rotary_encoder', 'BNC1High': 'reset2_rotary_encoder', 'BNC1Low': 'reset2_rotary_encoder'},
        )

        log.info('Precue OFF')

        # Reset rotary encoder (see above). Move on after brief delay (to avoid a race conditions in the bonsai flow).
        sma.add_state(
            state_name='reset2_rotary_encoder',
            state_timer=0.05,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={'Tup': 'delay_on'},
        )

        # DELAY EPOCH
        #delay is called from the dataframe
        sma.add_state(
            state_name='delay_on',
            state_timer=self.df_contingencies['delay'], 
            output_actions=[('delay duration', self.df_contingencies['delay'])], #is it necessary to call upon bonsai to show the gray background that we see during the cue presentation?
            state_change_conditions={'Tup': 'cue_on'},
        )

        log.info('Delay ON for ' + str(self.df_contingencies['delay']) + ' seconds')

        # CUE EPOCH
        sma.add_state(
            state_name='cue_on',
            state_timer=self.task_params.CUE_PRESENTATION_TIME,
            output_actions=[self.bpod.actions.bonsai_show_stim, ('cue presentation', self.task_params.CUE_PRESENTATION_TIME)],
            state_change_conditions={'Tup': 'reset2_rotary_encoder', 'BNC1High': 'reset2_rotary_encoder', 'BNC1Low': 'reset2_rotary_encoder'},
        )

        log.info('Cue ON for ' + str(self.task_params.CUE_PRESENTATION_TIME) + ' seconds')

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

        log.info('Trial is over, ITI ON for ' + str(self.task_params.ITI_DURATION) + ' seconds')

        return sma

#launches the task
if __name__ == '__main__':  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments(parents=[DNMSSession.extra_parser()])
    sess = DNMSSession(**kwargs)
    sess.run()
