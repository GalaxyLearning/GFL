# GFL Framework

---------------------------------------------------

English | [简体中文](./README_cn.md)

**Galaxy Federated Learning Framework(GFL)** is a decentralized federated learning framework based on blockchain. GFL builds a decentralized communication network based on Ethereum, and executes key operations that requires credibility in FL through smart contracts.

## Quick Start

### 1. System Envs Required

a) GFL only supports Python3， please make sure your python version is no less than `3.4`.

b) GFL is based on `Pytorch`, so, `torch>=1.4.0` and `torchvision>=0.5.0` is required before using GFL. [Pytorch installation tutorial](https://pytorch.org/get-started/locally/)

### 2. Install

```shell
pip install gfl_p
```

### 3. Usage

The available commands of GFL.

```
usage: GFL [-h] {init,app,attach} ...

optional arguments:
  -h, --help         show this help message and exit

actions:
  {init,app,attach}
    init             init gfl env
    run              startup gfl
    attach           connect to gfl node
```

Init GFL node in `datadir` directory.

```shell
python -m gfl_p init --home datadir
```

Start GFL node(start in standalone mode by default). If you need to open console when starting node, use the `--console`` argument.

```shell
python -m gfl_p run --home datadir
```

Open console for operating GFL node. The following three methods can be used to connect to the node started in the previous step.

```
python -m gfl attach						# connect to http://localhost:9434 in default
python -m gfl attach -H 127.0.0.1 -P 9434
python -m gfl attach --home datadir
```

## GFL base design

![image-20210903165315547](./assets/GFL-base-framework.png)

The GFL framework is divided into two parts:

**Job Generator**

> Used to create a job that can be executed in the GFL network. Developers can use the interface provided by GFL to generate a Job for various configuration parameters and distribute them to the network for training.

**Run-Time Network**

> Several running nodes build GFL's decentralized training network, and each GFL node is also a node in the blockchain. These nodes continuously process the jobs to be trained in the network according to user commands.

## GFL core arch

![image-20210903213928765](./assets/GFL-core-framework.png)

+ **Manager Layer**

  + The start/stop/status operation of node
  + Provide communication interface for nodes
  + Sync job

+ **Scheduler Layer**

  + Manage the execution process of each job
  + Synchronize parameter files among nodes
  + Schedule the execution order of multiple jobs on the node

+ **FL Layer**

  + Configure the running environment of the job
  + Perform training/aggregation tasks
  + Provide the interfaces of user-defined action
