import torch.nn as nn
import torch
import yaml
from models.mlae.utils import get_data_array, MultiTaskPM25Dataset, get_dataloader
# from utils import get_data_array, MultiTaskPM25Dataset, get_dataloader
from models.mlae.custom_lstm import LSTM


class LSTM_Block(nn.Module):
    def __init__(self, config, num_stations, device):
        super(LSTM_Block, self).__init__()
        self.config = config
        self.device = device
        self.input_len  = self.config['input_len']
        self.input_dim = num_stations
        self.hidden_size_1 =  self.config['mlae_model']['lstm']['hidden_size'][0]
        self.hidden_size_2 =  self.config['mlae_model']['lstm']['hidden_size'][1]
        self.lstm_1 = nn.LSTM(input_size =  self.input_dim , hidden_size = self.hidden_size_1, num_layers = 1, batch_first = True).to(self.device)
        self.lstm_2 = nn.LSTM(input_size =  self.hidden_size_1 , hidden_size = self.hidden_size_2, num_layers = 1, batch_first = True).to(self.device)
    def forward(self, x_pm):

        batch_size = x_pm.shape[0]
        #Initialize hidden states and cell states
        (h_1,c_1), (h_2,c_2) = self.init_hidden(batch_size)

        # Pass pm 2.5 data through the first lstm layers
        # x_pm: (batch_size, input_len, num_stations)
        # output_1: (batch_size, input_len, hidden_size_1)
        # self.lstm_1.flatten_parameters()
        output_1, (_,_) = self.lstm_1(x_pm, (h_1, c_1) )

        # Pass through second lstm
        # output_2: (batch_size, input_len, hidden_size_2)
        # self.lstm_2.flatten_parameters()
        output_2, (h_2,_) =  self.lstm_2(output_1, (h_2, c_2) )

        # print(output_2.shape)
        return output_2

    def init_hidden(self, batch_size):
        h_1 = torch.zeros(1, batch_size, self.hidden_size_1).to(self.device)
        c_1 = torch.zeros(1, batch_size, self.hidden_size_1).to(self.device)

        h_2 = torch.zeros(1, batch_size, self.hidden_size_2).to(self.device)
        c_2 = torch.zeros(1, batch_size, self.hidden_size_2).to(self.device)
        return (h_1, c_1), (h_2,c_2)

class AE_Block(nn.Module):

    def __init__(self, config, num_stations, device):
        super(AE_Block, self).__init__()
        self.config = config
        self.device = device
        self.num_stations = num_stations
        self.num_features = len(self.config['meteo_features'])
        self.input_len  = self.config['input_len']
        self.input_size = self.num_features * self.num_stations
        # print(self.input_size)

        self.struct_encoder = self.config['mlae_model']['ae']['encoder']
        self.struct_decoder = self.config['mlae_model']['ae']['decoder']

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.struct_encoder[0]).to(self.device),
            nn.ReLU(),
            nn.Linear(self.struct_encoder[0], self.struct_encoder[1]).to(self.device),
            nn.ReLU(),
            nn.Linear(self.struct_encoder[1], self.struct_encoder[2]).to(self.device),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.struct_decoder[0], self.struct_decoder[0]).to(self.device),
            nn.ReLU(),
            nn.Linear(self.struct_decoder[0], self.struct_decoder[1]).to(self.device),
            nn.ReLU(),
            nn.Linear(self.struct_decoder[1], self.struct_decoder[2]).to(self.device),
            nn.ReLU(),
            nn.Linear(self.struct_decoder[2], self.input_size).to(self.device)
            # nn.ReLU()
        )

    def forward(self, x_meteo):
        # x_meteo: (batch_size, input_len, num_stations * num_features)

        latent_space = self.encoder(x_meteo)
        # print(latent_space.shape)
        # print(latent_space)
        reconstruction = self.decoder(latent_space)
        # print(reconstruction.shape)
        return latent_space, reconstruction

class MLAE(nn.Module):

    def __init__(self, config, num_stations, device):
        super(MLAE, self).__init__()

        self.config = config
        self.num_stations = num_stations
        self.device = device
        self.input_len = self.config['input_len']
        self.output_len = self.config['output_len']

        self.lstm_block = LSTM_Block(self.config, self.num_stations, self.device)
        self.ae_block = AE_Block(self.config, self.num_stations, self.device)

        self.mix_block_size = self.config['mlae_model']['mix']

        self.mix_block = nn.Sequential(
            nn.Linear(64, self.mix_block_size[0]).to(self.device),
            nn.ReLU(),
            nn.Linear(self.mix_block_size[0], self.mix_block_size[1]).to(self.device),
            nn.ReLU()
        )

        self.multitask_size = self.config['mlae_model']['multitask_size']

        prediction_block = nn.Sequential(
            nn.Linear(self.input_len * self.mix_block_size[1], self.multitask_size).to(self.device),
            nn.ReLU(),
            nn.Linear(self.multitask_size, self.output_len).to(self.device)
        )

        self.multitask_predict_layer = (
            nn.ModuleList([prediction_block for i in range(self.num_stations)] )
        )

    def forward(self, x_pm, x_meteo):

        lstm_output = self.lstm_block(x_pm)
        ae_latent_space,_ = self.ae_block(x_meteo)
        # print(lstm_output.shape, ae_latent_space.shape)
        encoding = torch.cat((lstm_output, ae_latent_space), dim = -1)
        # encoding = torch.cat((lstm_output.squeeze(0), ae_latent_space), dim = -1)
        representation = self.mix_block(encoding)
        # print(representation.shape)

        # print(representation.shape)
        representation = representation.view(representation.shape[0], representation.shape[1] * representation.shape[2])
        # print(representation.shape)
        # print(representation.shape)
        predictions = []

        for i in range(self.num_stations):
            prediction_block = self.multitask_predict_layer[i]
            station_prediction = prediction_block(representation)
            station_prediction = station_prediction.unsqueeze(1)
            # print(station_prediction.shape)
            predictions.append(station_prediction)
            # print(station_prediction.shape)
        predictions = torch.cat(predictions, dim = 1)
        return predictions


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = {'num_input_station': 3, 'station_selection_strategy' : 'correlation'}
    with open("../../config/mlae.yml", encoding = 'utf-8') as f:
        config =   yaml.safe_load(f)

    (pm_array,meteo_array), location, list_k_stations, scaler = get_data_array(args = args, config = config )
    # print(meteo_array, meteo_array.shape)
    dataset = MultiTaskPM25Dataset(pm25_data = pm_array, meteo_data = meteo_array, config = config)
    train_loader, valid_loader, test_loader = get_dataloader(pm25_data=pm_array, meteo_data = meteo_array, args = args, config = config, train_ratio=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # lstm_block = LSTM_Block(config, num_stations=10,device = device)
    # ae_block = AE_Block(config, num_stations=10, device = device)
    model = MLAE(config, num_stations=10, device = device)
    for i,data in enumerate(train_loader):
        xpm, ypm, xmeteo = data
        # print(xmeteo.shape)
        xpm, ypm, xmeteo = xpm.to(device), ypm.to(device), xmeteo.to(device)
        # print(xpm.shape, ypm.shape, xmeteo.shape)
        # lstm_out = model.lstm_block(xpm)
        # print(lstm_out.shape)
        latent_space, reconstruction = model.ae_block(xmeteo)
        # print(reconstruction.shape)
        # print(lstm_out.shape, latent_space.shape)
        output = model(xpm, xmeteo)
        print(output.shape)
        # print(xmeteo.shape)
        # print(output.shape, ypm.shape)
        break
