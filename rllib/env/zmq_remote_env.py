from collections import OrderedDict, Iterable
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
from .zmq_remote_worker import zmq_worker

logger = logging.getLogger(__name__)
ZMQ_CONNECT_METHOD = 'tcp'


class WorkerError(BaseException):
    pass


class ZMQRemoteVectorEnv(BaseEnv):
    """Vector env that executes envs in another process using ZMQ.
    Waits for all envs to compute, but is generally very fast.
    Both single and multi-agent child envs are supported.
    """
    def __init__(self, make_env, num_envs, multiagent,
                 remote_env_batch_wait_ms):
        self.make_local_env = make_env
        self.num_envs = num_envs
        self.multiagent = multiagent
        self.poll_timeout = remote_env_batch_wait_ms / 1000

        self.processes = None  # lazy init

    def poll(self):
        # Init processes
        if self.processes is None:
            self._setup()
            obs, rewards, dones, infos = self._setup_reset()
            return obs, rewards, dones, infos, {}

        # each keyed by env_id in [0, num_remote_envs)
        obs, rewards, dones, infos = {}, {}, {}, {}

        # wait for all envs to finish
        for e_id, remote in self._zmq_sockets.items():
            result = remote.recv()
            self._check_for_errors(result, e_id)
            _, rew, done, info = json.loads(result.decode())
            obs[e_id] = self._obs_to_numpy(self.shared_memories[e_id])
            rewards[e_id] = rew
            dones[e_id] = done
            infos[e_id] = info

        self.waiting = False
        return obs, rewards, dones, infos, {}

    def send_actions(self, action_dict):
        for env_id, actions in action_dict.items():
            socket = self._zmq_sockets[env_id]
            a = {}
            for k, v in actions.items():
                # TODO: support nested action dict?
                if isinstance(v, np.ndarray):
                    a[k] = v.item()
                elif isinstance(v, Iterable):
                    a[k] = [x.item() for x in v]
                else:
                    a[k] = v.item()
            msg = json.dumps(a)
            socket.send(msg.encode(), zmq.NOBLOCK, copy=False, track=False)

        self.waiting = True

    def try_reset(self, env_id):
        # only reset the env
        self._zmq_sockets[env_id].send('reset'.encode())
        return ASYNC_RESET_RETURN

    def stop(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self._zmq_sockets:
                remote.recv()
        for socket in self._zmq_sockets:
            socket.send('close'.encode())
        for p in self.processes:
            p.join()
        self.closed = True

    def _setup(self):
        self.waiting = False
        self.closed = False
        self.processes = {}

        self._zmq_context = zmq.Context()
        self._zmq_sockets = {}

        # iterate envs to get torch shared memory through pipe then close it
        self.shared_memories = {}

        for w_ind in range(self.num_envs):
            pipe, w_pipe = mp.Pipe()
            socket, port = zmq_robust_bind_socket(self._zmq_context)

            process = mp.Process(target=zmq_worker, args=(
                w_pipe, pipe, port, CloudpickleWrapper((self.make_local_env, w_ind))
            ))
            process.daemon = True
            process.start()
            self.processes[w_ind] = process

            self._zmq_sockets[w_ind] = socket

            pipe.send(('get_shared_memory', None))
            self.shared_memories[w_ind] = pipe.recv()

            # switch to zmq socket and close pipes
            pipe.send(('switch_zmq', None))
            pipe.close()
            w_pipe.close()

        logger.info("All remote envs started")

    def _setup_reset(self):
        for _, socket in self._zmq_sockets.items():
            socket.send('reset'.encode())

        all_obs = {}
        all_rew = {}
        all_info = {}
        all_done = {}
        for r_ind, remote in self._zmq_sockets.items():
            result = remote.recv().decode()
            self._check_for_errors(result, r_ind)
            _, rew, done, info = json.loads(result)

            all_obs[r_ind] = self._obs_to_numpy(self.shared_memories[r_ind])
            # each keyed by agent_id in the env
            all_rew[r_ind] = rew
            all_info[r_ind] = info
            all_done[r_ind] = done

        return all_obs, all_rew, all_done, all_info

    @staticmethod
    def _obs_to_numpy(obs):
        o = OrderedDict({})
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                o[k] = v.numpy()
            # support double layer dict
            elif isinstance(v, dict):
                o[k] = OrderedDict({})
                for nk, nv in v.items():
                    if isinstance(nv, dict):
                        dd = OrderedDict({})
                        for dk, dv in nv.items():
                            if isinstance(dv, dict):
                                raise NotImplementedError('Double Nested obs space dict not implemented')
                            else:
                                dd[dk] = dv.numpy()
                        o[k][nk] = dd
                    else:
                        o[k][nk] = nv.numpy()
            else:
                raise NotImplementedError('Unsupported obs type {}'.format(type(v)))
        return o

    def _check_for_errors(self, result, e_id):
        if result[:5] == b'error':
            error = 'Worker {} has an error {}'.format(e_id, result)
            raise WorkerError(error)


class CloudpickleWrapper(object):
    """
    Modified.
    MIT License
    Copyright (c) 2017 OpenAI (http://openai.com)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


def zmq_robust_bind_socket(zmq_context):
    try_count = 0
    while try_count < 3:
        try:
            socket = zmq_context.socket(zmq.PAIR)
            port = np.random.randint(5000, 30000)
            if ZMQ_CONNECT_METHOD == 'tcp':
                socket.bind("tcp://*:{}".format(port))
            if ZMQ_CONNECT_METHOD == 'ipc':
                os.makedirs('/dev/shm/remotezmq/', exist_ok=True)
                socket.bind("ipc:///dev/shm/remotezmq/{}".format(port))
        except zmq.error.ZMQError as e:
            try_count += 1
            socket = None
            last_error = e
            continue
        break
    if socket is None:
        raise Exception("ZMQ couldn't bind socket after 3 tries. {}".format(last_error))
    return socket, port

