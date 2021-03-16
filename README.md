# DNLE: Dataset for Noise Level Estimation

The **DNLE dataset** contains 1668 environmental background noise recordings labeled according to type and level of noise.
The recordings are approximately equally balanced between three main categories, i.e., *"mechanic"*, *"melodic"*, and *"quiet"*. These noise categories were selected as we found in a previous study that these are the background noises that can distract users in crowdsourcing when performing tasks [[1]](#1) [[2]](#2).
The recordings were collected through the audio-web API employing different Windows and Mac computers.

The loudness measurements were 50.7dB(A) on average, varying from 30.6dB(A) to 81.3dB(A).


| Noise Classes | Noise Category           | Files per<br>Category  | Loudness Average<br>(dBA) | min / max <br> dBA |
| ------------- |:-------------:| -----:|--------:|--------:|
| street, traffic <br> dishwasher  | mechanic | 576 | 58.6 | 46.5 / 70.0 |
| TV, TV-Show <br> music, radio    | melodic  | 580 | 59.6 | 32.0 / 81.3 |
| quiet                            | quiet    | 512 | 31.4 | 30.6 / 33.0 |


## Repository content

- [`audio/*.wav`](audio/)

  1668 audio recordings in WAV format (15.0 seconds long on average, 48.0 kHz, stereo) with the following naming convention:
  
  `{NOISE_TYPE} _ {TIMESTAMP} _ {NOISE_LEVEL} _ {ID}.wav`
  
  - `{NOISE_TYPE}` - type of environmental background noise,
  - `{NOISE_LEVEL}` - noise level measured in dBA,
  
- [FeatureGenerator](FeatureGenerator/)

Code to extract different spectral and chroma features from the audio files. The output is saved as *.csv. The code is optimized to run parallel.

- [LevelEstimator](LevelEstimator/)

Code to train and test the model based on a "Long Short-Term Memory" (LSTM) architecture in PyTorch.


## Citing

If you find this dataset useful in an academic setting please cite:
(to be updated)


## Download

The dataset can be downloaded as a single .zip file (~2.38 GB):

**[Download DNLE dataset](https://depositonce.tu-berlin.de/bitstream/11303/12788/2/audios.zip)**


## References

<a id="1">[1]</a>
R. Zequeira Jiménez, B. Naderi, and S. Möller, "Background Environment Characteristics of Crowd-Workers from German Speaking Countries Experimental Survey on User Environment Characteristics," in 2019 Eleventh International Conference on Quality of Multimedia Experience (QoMEX), 2019, pp. 1–3. [DOI: https://doi.org/10.1109/QoMEX.2019.8743208]

<a id="2">[2]</a>
R. Zequeira Jiménez, B. Naderi, and S. Möller, "Effect of Environmental Noise in Speech Quality Assessment Studies using Crowdsourcing," in 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX), 2020, pp. 1–6. [DOI: https://doi.org/10.1109/QoMEX48832.2020.9123144]