#  Automated Essay Scoring repo
==============================

This repository contains code and resources for a Kaggle competition focused on training a model to score student essays automatically. The goal is to reduce grading time and costs, enabling essays to be included in testing, a crucial measure of student learning often avoided due to grading challenges.

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

