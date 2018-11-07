import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import shutil
import os

from actor import Actor, actor_process
from learner import Learner, learner_process

def run():
    mp.freeze_support()

    n_actors = 16
    workers = [Learner(n_actors)]
    for actor_id in range(n_actors):
        workers.append(Actor(actor_id))

    #learner = Learner(n_actors)
    processes = [mp.Process(target = learner_process, args=(n_actors,))]
    for actor_id in range(n_actors):
        processes.append(mp.Process(target = actor_process, args=(actor_id,)))
    for pi in range(len(processes)):
        processes[pi].start()
    for p in processes:
        p.join()

if __name__ == '__main__':
    run()