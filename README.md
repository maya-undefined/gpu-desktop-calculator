![License](https://img.shields.io/github/license/maya-undefined/gpu-desktop-calculator.svg) ![Version](https://img.shields.io/github/v/tag/maya-undefined/gpu-desktop-calculator.svg) ![Languages](https://img.shields.io/github/languages/top/maya-undefined/gpu-desktop-calculator.svg)

# gpu-desktop-calculator

This project is a GPU-accelerated calculator designed for high-performance computations. It supports basic arithmetic operations, trigonometric functions, and (eventually) statistical calculations such as mean, median, and standard deviation. The calculator is built using CUDA and is optimized for NVIDIA GPUs.

Perfect if you need to add two absurdly large columns of numbers from the cli.

Definitely work-in-progress. I eventually want to do more complex calculations you might find in AI

## Requirements
- NVIDIA GPU with CUDA support
- CUDA toolkit

## Installation
- Ensure you have a CUDA-compatible GPU and the CUDA toolkit installed.
- Clone the repository: `git clone https://github.com/maya-undefined/gpu-desktop-calculator`
- Navigate to the project directory: `cd gpu-desktop-calculator`
- Compile the project: `nvcc ./gdc.cu kernels.cu FH.cpp verb.cu -o gdc `

### install nvcc

	sudo apt install nvidia-cuda-toolkit nvidia-container-toolkit docker.io

### building

	nvcc FH.cpp gdc.cu kernels.cu  verb.cu -o gdc
 
### usage

Generally we use `gdc <verb> file1 file2 out.file`. 

	./gdc add f1.dat f2.dat out.dat
	./gdc exp f1.dat out.dat
	./gdc div f1.dat f2.dat out.dat

## Input files

Files should be organized in columns of numbers. `float` is supported for now. The code will typically operate across (axis=0)

	1 4.0
	2.0 5.0
	3.0 6.0

Files can be single or multiple column where the operation makes sense

A hypothetical usage to calculate a logistic regresion given a pre-trained w.dat might be

	./gpu mul w.dat x.dat - | ./gpu add bias.dat - | ./gpu mul -1 - | ./gpu exp - | ./gpu add 1 - | ./gpu div 1 yhat.dat 

	y^​=1+e−(w⋅x+b)1​

### data

Test data can be generated by `datagen.sh`. These will create 1.6 GB of test data.

### Initial Performance Results

			complexPhysicsCalculation	  addArrays  
	cpu 		75.9				  67.3
	gpu 		61.7				  61.8

	implementation vs average time in seconds for generated data

