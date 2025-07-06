# DFFormer
This repo holds code for [DFFormer: Capturing Dynamic Frequency Features to Locate Image
Manipulation through Adaptive Frequency Transformer and Prototype Learning]

## Usage

### 1. Download Google pre-trained models
* [Get models in this link](https://drive.google.com/drive/folders/1S1BJyFWw4Tlb_ItdtzL9J1TV_as9tIbt?usp=drive_link): DFFormer-tinny, DFFormer-small, DFFormer-large


### 2. Prepare data

Please download the IML-MUST dataset.<br>
* [Baidu Disk](https://pan.baidu.com/s/180TzwbTHj1Q3FOvIwT3vyg?pwd=gdit) <br>
* [Google Drive](https://drive.google.com/drive/folders/1bCCRaP7MKkEhxbTBbcKvy0AHBFi6ZMQQ?usp=drive_link)

### 3. Environment

Please prepare an environment with Python=3.8, Pytorch=1.10.1, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on the IML-MUST dataset.

```bash
#!/bin/bash 
torchrun --nproc_per_node=3 train.py
```

- Run the test script on the Coverage dataset.

```bash
python evaluation.py  --model_name  dfformer-large
```

### 5. CKPT
* [Google Dirve](https://drive.google.com/drive/folders/1S1BJyFWw4Tlb_ItdtzL9J1TV_as9tIbt?usp=drive_link)
* [Baudu Disk](https://pan.baidu.com/s/1x9SkoEO8-QWA7yquSgx1Ew?pwd=gdit)

## Citations

```bibtex

```
