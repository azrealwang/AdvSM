# README

## Overview

This project provides implementations for "Sample-specific Noise Injection For Diffusion-Based Adversarial Purification". The experiments are conducted on the CIFAR-10 and ImageNet datasets. The codebase allows for separate execution of attack generation and evaluation phases.

## Datasets

### CIFAR-10
- **Usage**: No need to download manually; the dataset will be automatically downloaded and prepared when running the code.

### ImageNet

- **Description**: ImageNet is a large-scale dataset with millions of images across 1,000 classes.
- **Required Version**: For this project, you need the **ImageNet LSVRC 2012 Training/Validation Set**.
- **Preparation Steps**:
  1. **Download** the ImageNet 2012 training/validation set from the official [ImageNet website](https://www.image-net.org/).
  2. **Convert to LMDB Format**:
     - Use provided dataset.py to convert the downloaded dataset into **LMDB** format for efficient data loading.
     - Ensure that the data is organized correctly to match the expected input format of the code.

## Diffusion Model

- **Description**: The project utilizes a diffusion model for adversarial purification and generation tasks.
- **Download Link**: [https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view?usp=sharing](CIFAR-10)
[https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt](ImageNet)

  *(Note: Please insert the appropriate download link for the diffusion model.)*

## Running Experiments

Experiments can be run using the `main.py` script. The codebase is designed to separate the attack generation phase from the evaluation phase.

### Example Script

Below is an example of how to run an experiment for generating adversarial examples using PGD+EOT attack:

```bash
for seed in $SEED1; do
    python -u main.py --config cifar10.yml \
        --defense_method diffpure \
        --dataset cifar10 \
        --classifier_name cifar10-wideresnet-28-10 \
        --attack_type pgd \
        --seed $seed \
        --batch_size 16 \
        --n_iter 200 \
        --eot 20 \
        --att_max_timesteps 100 \
        --att_num_denoising_steps 20 \
        --att_sampling_method ddpm \
        --num_process_per_node 2 \
        --use_cuda \
        --port 8888
done
```
Below is an example of how to run an experiment for evaluation:

```bash
for seed in $SEED1; do
    python -u main.py --config cifar10.yml \
        --defense_method diffpure \
        --dataset cifar10 \
        --classifier_name cifar10-wideresnet-28-10 \
        --attack_type pgd \
        --seed $seed \
        --batch_size 32 \
        --n_iter 200 \
        --eot 20 \
        --att_max_timesteps 100 \
        --att_num_denoising_steps 20 \
        --att_sampling_method ddpm \
        --def_max_timesteps 100 \
        --def_num_denoising_steps 100 \
        --def_sampling_method ddpm \
        --num_process_per_node 1 \
        --use_cuda \
        --whitebox_defense_eval \
        --port 8888
done
```

Below is an example of how to run an experiment for generating adversarial examples using PGD+EOT attack on ImageNet:

```bash
for seed in $SEED1; do
    python -u main.py --config imagenet.yml \
        --defense_method diffpure \
        --dataset imagenet \
        --classifier_name imagenet-resnet50 \
        --attack_type pgd \
        --seed $seed \
        --batch_size 1 \
        --n_iter 200 \
        --eot 20 \
        --att_max_timesteps 200 \
        --att_num_denoising_steps 10 \
        --att_sampling_method ddpm \
        --num_process_per_node 4 \
        --use_cuda \
        --port 8888
done
```
The evaluation is the same as CIFAR-10 above.
You can reproduce our results using seed: 121, 122, 123.

### Evaluation and Attack Phases

- **Attack Generation**:
  - Before evaluating the defense mechanisms, adversarial examples need to be generated.
  - Use the appropriate attack parameters to create adversarial samples.

- **Evaluation Activation**:
  - To activate the evaluation phase, include the following arguments in your command:
    - For standard white-box defense evaluation:
      ```bash
      --whitebox_defense_eval
      ```
    - For adaptive defense evaluation:
      ```bash
      --adaptive_defense_eval
      ```
  - These flags will trigger the evaluation procedures after adversarial examples have been generated.

## Dependencies

- Python 3.7 or higher
- PyTorch 2.0.1
- Other dependencies as listed in `requirements.txt`

## Reference
Our codebase is implemented based on the following code, we would like to thank for their great work.
[https://github.com/NVlabs/DiffPure.git]
[https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification.git]
[https://github.com/ZSHsh98/EPS-AD.git]

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.