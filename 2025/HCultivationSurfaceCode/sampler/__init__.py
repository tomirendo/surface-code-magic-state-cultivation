
import sinter
from .sinter_sampler import PostSelectionSampler, PymatchingGapSampler 
import sys
sys.path.append("../StateVecSimulator")
sys.path.append("../StateVecSimulator/latte")

from _vec_intercept_sampler import VecInterceptSampler

def sinter_samplers() -> dict[str, sinter.Sampler]:
    return {"StateVecSimulatorUnrotatedd3" : VecInterceptSampler(logical_x = [0,1,2], logical_z = [0,5,10]),
            "PostSelectionSampler" : PostSelectionSampler(),
            "PymatchingGapSampler" : PymatchingGapSampler(),
            }

