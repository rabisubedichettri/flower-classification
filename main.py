
import os
from utils.config_loader import load_config,get_base_dic

config_loc=os.path.join(get_base_dic(),"configs","network.json")
config=load_config(config_loc)
if config["hardware"]["GPU"]["active"]:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from model.custom_model import BridNet
obj=BridNet(config).train()