The [Argus](https://github.com/lRomul/argus) framework for PyTorch was employed. It makes the learning process more straightforward and the code briefer.

## Quick setup and start 

### Requirements 

#### Software

* Linux
* Nvidia drivers, CUDA >= 11.3, cuDNN >= 8
* [Docker](https://www.docker.com), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

#### Hardware

* 32GB of RAM
* 2080ti or another GPU with fp16 support and at least 12GB memory 

### Preparations 

* Clone the repo, build docker image. 
    ```bash
    git clone https://github.com/5Hyeons/noise_detector.git
    cd noise_detector
    make build
    ```
* Download pretrained model in [here](https://drive.google.com/file/d/1VZ6Mygt7JAYIeF2Nay_gEHHdFctjUIqv/view?usp=share_link) and unzip into the 'data/experiments' folder
    ```bash
    unzip experiments.zip -d data/experiments
    ```
* Copy the Makefile to the path where you want to use the package.
    ```bash
    cp Makefile '../'
    cd ..
    ```
* Run docker container 
    ```bash
    make run
    ```

### Usage
* Set variables   
    ```python
    import noise_detector

    data_path = 'your_data_path'
    output_path = 'output_csv_path_what_you_want'
    device = 'cuda'
    ```
    * data_path : Directory containing audio files
    * output_path : File path to save    
    * device : If you want to run on CPU, set 'cpu' 
   

* Auto labeling whether It is noisy or not 
    ```python
    noise_detector.predict(data_path=data_path, output_path=output_path, device=device)
    ```
    * output sample
        |fname|state|
        |------|---|
        |p_232_001.wav|Noisy|
        |p_232_002.wav|Noisy|
        |p_232_003.wav|Clean|
        |p_232_005.wav|Clean|
* Evaluate speech quality 
    ```python
    noise_detector.DNSMOS.quality_evaluate(data_path=data_path, output_path=output_path, personalized_MOS=False, only_mos=True)
    ```
    * output sample [only_mos=True]
        |fname|len_in_sec|sr|num_hops|P808_MOS|
        |------|---|---|---|---|
        |test/p232_001.wav|1.7413125|16000|4|3.3465686|
        |test/p232_002.wav|2.71525|16000|1|3.5216465|
        |test/p232_022.wav|5.794|16000|2|4.1341343|
        |test/p232_023.wav|9.768875|16000|1|4.0300155|
    * output sample [only_mos=False]
        |fname|len_in_sec|sr|num_hops|OVRL_raw|SIG_raw|BAK_raw|OVRL|SIG|BAK|P808_MOS|
        |------|---|---|---|---|---|---|---|---|---|---|
        |test/p232_001.wav|1.7413125|16000|4|3.6834536|4.136113|3.9792447|3.236603143751564|3.6181635989500043|3.9217443921938164|3.3465686|
        |test/p232_002.wav|2.71525|16000|1|3.8041916|4.3243375|3.9037356|3.3102587774176055|3.7142838991512788|3.879148739937948|3.5216465|
        |test/p232_022.wav|5.794|16000|2|4.0243034|4.3224835|4.2989554|3.4391541205959113|3.713366195103027|4.0879562307229795|4.1341343|
        |test/p232_023.wav|9.768875|16000|1|4.135815|4.369508|4.411438|3.5020105980389493|3.736453177555213|4.14026287237692|4.0300155|
   

* Run both at the same time

    ```python
    noise_detector.predict(data_path=data_path, output_path=output_path, device=device)
    noise_detector.DNSMOS.quality_evaluate(data_path=data_path, output_path=output_path, personalized_MOS=False, only_mos=True)
    ```
    * output sample
        |fname|state|len_in_sec|sr|num_hops|P808_MOS|
        |------|---|---|---|---|---|
        |test/p232_001.wav|Noisy|1.7413125|16000|4|3.3465686|
        |test/p232_002.wav|Noisy|2.71525|16000|1|3.5216465|
        |test/p232_022.wav|Speech|5.794|16000|2|4.1341343|
        |test/p232_023.wav|Speech|9.768875|16000|1|4.0300155|

## References

[1] [argus-freesound](https://github.com/lRomul/argus-freesound): Kaggle | 1st place for Freesound Audio Tagging 2019.

[2] [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS): A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors.

[3] [Noisy speech database](https://datashare.ed.ac.uk/handle/10283/2791) for training speech enhancement algorithms and TTS models. 