# HumanEval-Rel

## Introduction

This repository aims at centralizing the following elements:

* **a dataset**: an evaluation dataset to benchmark LLMs on code generation for reliability engineering.
* **source code**: an assessment framework to evaluate LLMs on the evaluation dataset.

The dataset and source code will follow the framework from [HumanEval, introduced in Chen et al. (2021)](https://arxiv.org/abs/2107.03374).

## TODO

- [ ]  Change the code so that it supports environments that do not include CUDA and are CPU-only.

## Quickstart

To get started with the current code, you will need the `make` command to setup the project.
Note: the current code assumes that the environment supports CUDA.

### 1 - Installing `make` (Windows)

To check if you have `make` already installed, you can run:

```bash
make --version
```

If it is not installed, you can use `chocolatey` to install it.

To check if you have `chocolatey` already installed, you can run:

```bash
choco --version
```

If it is not already installed, you can install `chocolatey` following [this quick tutorial](https://chocolatey.org/install).

Then, once `chocolatey` is setup, you can install `make` by running:

```bash
choco install make
```

### 2 - Setting up the project

Then, once `make` is installed, you can setup the project by running:

```bash
make all
```

### 3 - Running the code

Once the project has been set up a first time, you can run the code at any point by running:

```bash
make run
```