DR-DDP
========
This repository containts the official Python implementation of **DR-DDP** algorithm presented in the paper **[Distributionally Robust Differential Dynamic Programming with Wasserstein Distance](https://arxiv.org)**.

## Requirements

To run the code in this repository, make sure you have the following dependencies installed:

-   Python (>= 3.6)
-   Gym
-   PyTorch
-   SciPy
-   Other standard Python libraries, such as NumPy, Matplotlib, and Pickle.

## Installation

Follow these steps to install DR-DDP and its dependencies:

1.  Clone the repository:
```
$ git clone https://github.com/CORE-SNU/DR-DDP.git
```

2.  Navigate to the DDP Systems folder:
```
$ cd DR-DDP/DDP_Systems
```

3.  Install the Gym environments:
```
$ pip install -e .
``` 


## Usage

In the main DR-DDP directory, run the `main.py` code with the desired arguments:
```
$ python main.py --dist <distribution> --num_sim <num_simulations> --num_samples <num_samples> --system <system_type> --baseline <baseline_type> --plot --render
``` 

Here's an explanation of each argument:

 -   `--dist <distribution>`: Specify the disturbance distribution to use. Replace `<distribution>` with the desired distribution name (either `normal` or `uniform`).
 -   `--num_sim <num_simulations>`: Set the number of simulations to run. Replace `<num_simulations>` with the desired number.
 -   `--num_samples <num_samples>`: Set the number of samples to generate in each simulation. Replace `<num_samples>` with the desired number.
 -   `--system <system_type>`: Specify the system to be controller. Replace `<system_type>` with the desired system name (either `Oscillator-v0` or `Car-v0`)
 -    `--baseline <baseline_type>`: Specify the type of baseline controller. Replace `<baseline_type>` with the desired baseline type (`1` to use GT-DDP, `2` to use both GT-DDP and box-DDP).
 -   `--plot`: Include this flag to enable plotting of the simulation results.
 -   `--render`: Include this flag to enable rendering of the system.


## Example

To apply the DR-DDP algorithm for a kinematic car navigation problem, run the following command:
```
$ python main.py --dist uniform --num_sim 1 --num_samples 10 --system Car-v0 --baselin 2 --plot
```

The following figure demonstrates an example of the resulting trajectories:

<img src="https://github.com/CORE-SNU/DR-DDP/blob/main/trajs.png"  width="40%" height="40%">
