
import os
from utils.config_loader import load_config,get_base_dic
from model.custom_model import BridNet

config_loc=os.path.join(get_base_dic(),"configs","network.json")
config=load_config(config_loc)
obj=BridNet(config).train()