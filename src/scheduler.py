import numpy as np


class TCScheduler:
    '''
    Transaction cost scheduler during training:
    TC_train(x) = (TC_max / S^a) * x^a  for x <= S
    TC_train(x) = TC_max               for x > S
    
    where:
      TC_max = 0.0025
      S = 100 * episode_length
      a = 1 for Phase 1, a = 0.45 for Phases 2 and 3
    '''
    def __init__(self, episode_length: int, phase: int = 1,
                 tc_max: float = 0.0025):
        self.tc_max = tc_max
        self.episode_length = episode_length
        self.S = 100 * episode_length
        # Paper: a=1 for Phase 1, a=0.45 for Phases 2 and 3
        self.a = 1.0 if phase == 1 else 0.45
        self._global_step = 0

    def get_tc(self, local_step: int = None) -> float:
        '''Get TC at current global step.'''
        x = self._global_step
        if x <= self.S:
            tc = (self.tc_max / (self.S ** self.a)) * (x ** self.a)
        else:
            tc = self.tc_max
        return float(np.clip(tc, 0.0, self.tc_max))

    def step(self):
        '''Increment global step counter.'''
        self._global_step += 1

    def reset_for_phase(self, phase: int):
        '''Reset and set convexity for new phase.'''
        self.a = 1.0 if phase == 1 else 0.45
        self._global_step = 0
