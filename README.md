# neuron_task Readme

Here we will introduce the task and the corresponding dataset information.

## Task Definition

### Classification

Given a input $X \in \mathbb{R}^{N \times T}$ , where $N$ represents the neurons(variates) and $T$ represents the length of the input series, the target is to predict  the grid id that the mouse stays in $ Y \in\mathbb{R}^{1}$ , which is corresponding to the last point of the input series.

For example, the input is $X_1 =[x_1,x_2,x_3,x_4] =[1.0,2.0,3.0,4.0] \in \mathbb{R}^{1 \times 4} $ , where $N = 1$ and $T = 4$.  The target position is  $Y_1 = 10 \in \mathbb{R}^{1 }$ , which is corresponding to $x_4$. 

### Regression

Given a input $X \in \mathbb{R}^{N \times T}$ , where $N$ represents the neurons(variates) and $T$ represents the length of the input series, the target is to predict the position of the mouse  $\mathbb{R}^{1 \times 2}$ , which is corresponding to the last point of the input series.

For example, the input is $X_1 =[x_1,x_2,x_3,x_4]= [1.0,2.0,3.0,4.0] \in \mathbb{R}^{1 \times 4} $ , where $N = 1$ and $T = 4$. The target position is  $[1.0,2.0] \in \mathbb{R}^{1 \times 2}$ , which is corresponding to $x_4$. 

[Prediction length]：30（when given X_1 =[x_1,x_2,x_3,x_4]，model should predict target position corresponding from x_4 to x_33）

## Dataset

### Overall Information

- split ratio: [0.7,0.1,0.2] (train,valid,test) Please split the overall data series according to this split ratio.
- sample generation method: rolling window

### Classification

You can find the dataset in the subfolder "./classification".

- Number of classes:25
- Total length: 26986
- Number of variates(number of neurons) 147
- X: neuron_signals_aligned.npy 
- Y: grid_sequence.npy

### Regression

You can find the dataset in the subfolder "./regression".

- Total length: 26986
- Number of variates(number of neurons) 147
- Position series dimension: 2
- X: neuron_signals_aligned.npy 
- Y: positions_cm.npy



