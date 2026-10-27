# MLDS 490: Advanced Algorithms in Artifical Intelligence
By Anuraj Nair

## Homework 1
This assignment comes in two parts. 

Part 1 focuses on tabular solutions for a shuttle servicing problem. 

Part 2 focuses on policy gradient solutions for CartPole-v1 and Pong-v5, provided by Gymnasium.

## Directory Structure
```
.
├── activate_conda_env.txt
├── config.yaml
├── notebooks
│   └── iterative_methods.ipynb
├── README.md
├── requirements.txt
├── scripts
│   └── runner_pg.sh
└── src
    ├── q1
    │   ├── edge_cases_validation_test.py
    │   ├── enumeration_methods.py
    │   ├── figures
    │   │   ├── part_a_enumeration_V0.png
    │   │   ├── part_b_value_iteration_V.png
    │   │   ├── part_c_policy_iteration_policy.png
    │   │   ├── v3_threshold_sweep.png
    │   │   ├── v6_comparative_statics.png
    │   │   └── v7_convergence.png
    │   ├── init.py
    │   ├── notes_q1.md
    |   |-- tabular_eval_dictionary.md # Dictionary defining each validation and edge case test which occurs
    └── q2
        ├── cartpole_baseline.py
        ├── cartpole.py
        ├── init.py
        ├── pong_baseline.py
        ├── pong.py
        ├── results.md
        └── validation_edge_cases
            ├── val_common.py
            ├── validation_cartpole.py
            ├── validation_pong.py
            └── cartpole_pong_eval_dictionary.md  # Dictionary defining terms for the evaluation of Cartpole and Pong
            
```

## Set Up
1. Pull this repo to your local machine or into DeepDish.
2. We must first set up a conda environment. Using `cd` commands, navigate into the local location of this repo. Then, run `conda create --prefix ./.venv python=3.12`.
3. Activate this virtual environment inside the repo directory, `conda activate ./.venv`.
4. Install the requirements, `pip install -r requirements.txt`.
5. Create a jupyter notebook kernel using `ipython kernel install --user --name=<any name for kernel>`. 

## Instructions for Tabular Methods
1. To modify the parameters of these methods, you may edit `config.yaml`.
2. The methods for the shuttling services problem is found in `src/q1/enumeration_methods.py`. These methods are then ran, to demonstrate outputs, in the notebook `notebooks/iterative_methods.ipynb`.
3. To run the edge cases and validation test, run from the root of the repo `python3 src/q1/edge_cases_validation.py`. This will create a `figures` folder in your root directory where the results of the evaluation test will be stored. The CLI will demonstrate the results of the individual test with a description of the test as well, with further description in the docstring of the methods found in this file. 

## Instructions for Policy Gradient Models
1. The following instructions will be ran from the root directory.
2. If you wish to change the training hyperparameters, you may modify `config.yaml`.
3. The first part is to train the models.
   1. To train the non baseline CartPole, `python3 src/q2/cartpole.py`
   2. To train the CartPole Algorithm with a Baseline, `python3 src/q2/cartpole_baseline.py`
   3. To train the non baseline Pong model, `python3 src/q2/pong.py`.
   4. To train the Pong model with a baseline, `python3 src/q2/pong_baseline.py`.
4. [WARNING] Training the above models will take some time but will output the mean and std at the end of the training run. The model weights will be stored under the `models` directory in the root. 
5. To run evaluations on the models, run the shell script: `bash scripts/runner_pg.sh`. These evaluations will stored in the logs folder in your root repository, allowing you to examine the outputs. The evaluations are given a brief explanation of what they measure. Reference the respective code for a doc string explaining the test.
6. The evals will also output figures which can be examined underneath the `figures` directory from your root. 

## Definitions
To find definitions for each evaluation (edge cases and validation), refer to the respective dictionary md in each directory in `src`. These dictionaries explain what each test does and why it is done. Location of each dictionary can be pinned down in the directory tree above for further reference. 



