# Value_Iteration_for_Learning_Concurrently_Executable_Robotic_Control_Tasks
Supplementary code repository for 2025 AAMAS paper "Value Iteration for Concurrently Executable Robotic Control Tasks"

algos -> Contains implementation of our proposed version of continuous fitted value iteration for concurrently executable tasks as well as the min-norm controller that executes each of the learned value functions as CLFs.

system_models-> Contains abstract class for control affine systems required for code in algos. The model of single-integrator multi-robot systems used in the paper's experiments is provided. These models are not only compatible with our code, but are also valid Gymnasium environments, so can also be tested with existing RL libraries such as Stable Baselines.

This code was taken from and put together from my own private repositories that I used to conduct the research. There may be bugs, both in general and from me moving things around. You can direct any questions or comments to sheikh.abrar.tahmid@uwaterloo.ca. 

