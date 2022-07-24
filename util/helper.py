import numpy as np
import torch
import random, os

from models.mlae.supervisor import MultitaskLSTMAutoencSupervisor
from models.lstm.supervisor import LSTMSupervisor
from models.spattrnn.supervisor import SAERSupervisor
from models.ac_lstm.supervisor import AcLSTMSupervisor
from models.cnn_lstm.supervisor import CNNLSTMSupervisor
from models.encoder_decoder.supervisor import EDLSTMSupervisor
from models.gc_lstm.supervisor import GCLSTMSupervisor
from models.geoman.supervisor import GEOMANSupervisor
from models.imda_vae.supervisor import ImdaVAESupervisor
from models.magan.supervisor import MAGANSupervisor
from models.daqff.supervisor import DAQFFSupervisor

def get_config(model_type):
    config_path = 'config/'
    return config_path + model_type + '.yml'

def model_mapping(model_type):
    config_path = 'config/'
    if model_type == 'mlae':
        res = {
            'model': MultitaskLSTMAutoencSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'lstm':
        res = {
            'model': LSTMSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'spattrnn':
        res = {
            'model': SAERSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'encoder_decoder':
        res = {
            'model': EDLSTMSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'ac_lstm':
        res = {
            'model': AcLSTMSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'cnn_lstm':
        res = {
            'model': CNNLSTMSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'daqff':
        res = {
            'model': DAQFFSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'gc_lstm':
        res = {
            'model': GCLSTMSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'geoman':
        res = {
            'model': GEOMANSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'imda_vae':
        res = {
            'model': ImdaVAESupervisor,
            'config': config_path + model_type + '.yml'
      }
    elif model_type == 'magan':
        res = {
            'model': MAGANSupervisor,
            'config': config_path + model_type + '.yml'
        }
    
    return res 

def seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True