from .base01 import BaseModel_01
from .base02 import BaseModel_02
from .base03 import BaseModel_03

from .heart01 import HeartNetwork_01
from .heart02 import HeartNetwork_02

from .self01 import SelfModel_01
from .self02 import SelfModel_02
from .self03 import SelfModel_03
from .self04 import SelfModel_04

MODEL_REGISTRY = {
    "base01": BaseModel_01,
    "base02": BaseModel_02,
    "base03": BaseModel_03,
    #
    "heart01": HeartNetwork_01,
    "heart02": HeartNetwork_02,
    #
    "self01": SelfModel_01,
    "self02": SelfModel_02,
    "self03": SelfModel_03,
    "self04": SelfModel_04,
}


def build_model(config):
    model_name = config["model"]["name"]
    in_channels = config["dataset"]["in_channels"]
    out_channels = config["dataset"]["out_channels"]
    model_class = MODEL_REGISTRY[model_name]
    kwargs = {"in_channels": in_channels, "out_channels": out_channels}

    return model_class(**kwargs)
