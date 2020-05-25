import gym
import json
import os
import logging
import pickle

import cloudpickle
import numpy as np
import torch
import zmq
from torch import multiprocessing as mp

import ray
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN

logger = logging.getLogger(__name__)
ZMQ_CONNECT_METHOD = 'ipc'


def zmq_worker(remote, parent_remote, port, env_fn_wrapper):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """
    parent_remote.close()
    env_fn, env_id = env_fn_wrapper.x
    env = env_fn(env_id)

    shared_memory = {}
    for name, space in env.observation_space.spaces.items():
        if isinstance(space, gym.spaces.Box):
            if space.dtype != np.float32:
                raise NotImplementedError('Type not implemented {}'.format(space.dtype))
            tensor = torch.zeros(space.shape, dtype=torch.float32)
        elif isinstance(space, gym.spaces.dict.Dict):
            raise NotImplementedError('Gym dict spaces not supported')
        shared_memory[name] = tensor

    # initial python pipe setup
    python_pipe = True
    while python_pipe:
        cmd, _ = remote.recv()
        if cmd == 'get_shared_memory':
            remote.send(shared_memory)
        elif cmd == 'switch_zmq':
            # close python pipes
            remote.close()
            python_pipe = False
        else:
            raise NotImplementedError

    # zmq setup
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    if ZMQ_CONNECT_METHOD == 'tcp':
        socket.connect("tcp://localhost:{}".format(port))
    if ZMQ_CONNECT_METHOD == 'ipc':
        socket.connect("ipc:///dev/shm/remotezmq/{}".format(port))

    running = True
    while running:
        try:
            socket_data = socket.recv()
            socket_parsed = socket_data.decode()

            # commands that aren't action dictionaries
            if socket_parsed == 'reset':
                ob = env.reset()
                # MUST return ob, reward, done, info
                # TODO: should be different for not multi agent env
                reward = {agent_id: 0 for agent_id in ob.keys()}
                done = {"__all__": False}
                info = {agent_id: {} for agent_id in ob.keys()}
                # only the non-shared obs are returned here
                non_shared_ob = handle_ob(ob, shared_memory)
                msg = json.dumps((non_shared_ob, reward, done, info))
                socket.send(msg.encode(), zmq.NOBLOCK, copy=False, track=False)
            elif socket_parsed == 'close':
                env.close()
                running = False
            # else action dictionary
            else:
                action_dictionary = json.loads(socket_parsed)
                ob, reward, done, info = env.step(action_dictionary)
                # Done ob handled by reset
                # if isinstance(done, dict):
                    # if done['__all__']:
                        # ob = env.reset()
                # elif done:
                    # ob = env.reset()
                ob = handle_ob(ob, shared_memory)
                # only the non-shared obs are returned here
                msg = json.dumps((ob, reward, done, info))
                socket.send(msg.encode(), zmq.NOBLOCK, copy=False, track=False)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            running = False
            e_str = '{}: {}'.format(type(e).__name__, e)
            print('Subprocess environment has an error', e_str)
            socket.send('error. {}'.format(e_str).encode(), zmq.NOBLOCK, copy=False, track=False)


def handle_ob(ob, shared_memory):
    non_shared = {}
    for k, v in ob.items():
        if isinstance(v, torch.Tensor):
            shared_memory[k].copy_(v)
        elif isinstance(v, np.ndarray):
            shared_memory[k].copy_(torch.from_numpy(v))
        # support double layer dict
        elif isinstance(v, dict):
            for nk, nv in v.items():
                if isinstance(nv, dict):
                    raise NotImplementedError('Nested obs space dict not implemented')
                shared_memory[k][nk] = torch.from_numpy(nv)
        else:
            raise NotImplementedError('Unsupported obs type {}'.format(type(v)))
            # non_shared[k] = v
    return non_shared
