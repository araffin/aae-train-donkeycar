# Augmented Auto-Encoder Training Code

Code used to train an augmented auto-encoder (aka denoising auto-encoder with more augmentations) for the DonkeyCar simulator.

Presentation: [Learning To Race in Hours](https://araffin.github.io/talk/learning-race/)

![Augmented Auto-Encoder](https://araffin.github.io/slides/rlvs-tips-tricks/images/car/race_auto_encoder.png)

## Record data

1. [Download](https://github.com/tawnkramer/gym-donkeycar/releases) and launch the donkey car simulator

2. Install dependencies
```
# Install current package
pip install -e .
# If not using custom donkey car gym
pip install git+https://github.com/araffin/gym-donkeycar-1@feat/live-twitch
```

3. Drive around randomly (make sure to check the script first)

```
python record_data.py --max-steps 10000 -f logs/dataset-mountain
```

## Train the AutoEncoder

0. [Optional, only a folder with images is required] Split video into a sequence of images
```
python -m ae.split_video -i logs/videos/video.mp4 -o logs/dataset/
```

1. Train the autoencoder (with data-augmentation)
```
python -m ae.train_ae --n-epochs 100 --batch-size 8 --z-size 32 -f logs/dataset-test/ --verbose 0

# You can train on multiple datasets easily:
python -m ae.train_ae --n-epochs 200 --z-size 32 -f logs/dataset-0/ logs/dataset-1/ --batch-size 4
```

2. Have a coffee while the autoencoder is training ;)


3. [Optional but recommended] Inspect the trained autoencoder

```
python -m ae.test -f logs/dataset-test/ -ae logs/ae-32_000000.pkl --n-samples 50 -augment
```


## Use the AutoEncoder with a Gym wrapper

The Gym wrapper is `ae.wrapper.AutoencoderWrapper`, you can add it to the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) (branch "offline-rl").

```
# Export path to trained autoencoder
export AAE_PATH=/absolute/path/to/autoencoder.pkl
# Then you can call python train.py --algo ... --env ... with the RL Zoo
```
