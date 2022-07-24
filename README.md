# Air Quality Prediction Benchmark

This is Pytorch implementation of baselines in air quality prediction domain with the following paper:
1. LSTM: models/lstm
2. CNN-LSTM: models/cnn-lstm
3. GA-Encoder-Decoder: models/encoder-decoder
4. GC-LSTM: models/gc-lstm
5. SpAttRNN: models/spattrnn
6. AC-LSTM: models/ac-lstm
7. GeoMAN: models/geoman
8. MAGAN: models/magan
9. IMDA-VAE: models/imda-vae
10. DAQFF: models/daqff
11. Multitask LSTM Autoencoder: models/mlae


## Note to run wandb sweep
Follow https://docs.wandb.ai/guides/sweeps/quickstart

0. Define cac tham so can chay trong wandb_config 
1. Tao sweep instance wandb sweep link_to_wandb_config.yaml
2. main_with_args.py: Them cac tham so can tinh chinh trong arg_parse
3. supervisor.py: Sua config tham khao theo lstm/supervisor.py
    - Sua cac config 
    - Tai ham train, return val_loss cuoi cung
4. Chay command de run sweep
7. Vao link de xem ket qua chay

## Run code:
LSTM: 
    Train: python main.py --model=lstm --input_len=48 --output_len=1 --train_ratio=1 --target_station=all 

Encoder-Decoder
    Train: python main.py --model=encoder_decoder --input_len=48 --output_len=1 --train_ratio=1 --target_station=all 

SAER: 
    - ID: ja5tloix
    - Command: wandb agent aiotlab/Air-Quality-Prediction-Benchmark/ja5tloix
    - Link: https://wandb.ai/aiotlab/Air-Quality-Prediction-Benchmark/sweeps/ja5tloix
