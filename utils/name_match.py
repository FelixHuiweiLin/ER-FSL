from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from agents.exp_replay import ExperienceReplay
from agents.er_fsl import ER_FSL
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update

data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'mini_imagenet': Mini_ImageNet
}

agents = {
    'ER': ExperienceReplay,
    'ERFSL': ER_FSL
}

retrieve_methods = {
    'random': Random_retrieve,
}

update_methods = {
    'random': Reservoir_update,
}

