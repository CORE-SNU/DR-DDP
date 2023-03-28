#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.registration import register


register(
    id='Car-v0',
    entry_point='DDP_Systems.envs:RacingCar',
)


register(
    id='Oscillator-v0',
    entry_point='DDP_Systems.envs:Oscillator'
)


