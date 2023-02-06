import logging
import iblrig.misc
from iblrig.base_choice_world import BiasedChoiceWorldSession
from iblrig_tasks._iblrig_tasks_biasedChoiceWorld.task import run as run_biased
log = logging.getLogger("iblrig")


class Session(BiasedChoiceWorldSession):
    def draw_quiescent_period(self):
        """
        For this task we double the quiescence period texp draw and remove the absolute
        offset of 200ms. The resulting is a truncated exp distribution between 400ms and 1 sec
        """
        return iblrig.misc.texp(factor=0.35 * 2, min_=0.2 * 2, max_=0.5 * 2)


def run(*args, **kwargs):
    run_biased(sess=Session(*args, **kwargs))
