#  Automated Essay Scoring System
==============================


## Description

1. technical problem: automate the evaluating phase in essay questions.  
2. buissnes challenge: The goal is to reduce grading time and costs, enabling essays to be included in testing, a crucial measure of student learning often avoided due to grading challenges.

3. i have third approaches:
    - using a classical models with some heavey and smart feature engineering ( Done )

    - using LLMs, fine-tune in the dataset and some expermentations in the best and cost, tests in things like LoRa or full or some specific layers to be fine-tuned for fine-tune, like how to trate the problem in a classification way with 6 labels or regression task, and more ( working on it )

    - useing the feature from the first approache and using a LLM as a feature extraction then pass all to classic model ( i had made a notebook as a prove of concept )  


## requirements

- python 3.11 or later

### Install Python using Anaconda

1) Download and install Anaconda from [here](https://docs.anaconda.com/free/anaconda/install/index.html)
2) Create a new environment using the following command:
```bash
$ conda create -n essay-scoring python=3.11
```
3) activate the environment:
```bash
$ conda activate essay-scoring
```
## (Optional) Setup you command line interface for better readability

```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### setup the environment 

```bash
$ cp .env.example .env
```

### Note:
    all notebooks was tested, runned and experimented in Kaggle.


