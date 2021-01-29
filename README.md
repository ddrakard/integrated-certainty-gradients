# Integrated Certainty Gradients

## Introduction

This project demonstrates *Integrated Certainty Gradients*.
This is a new method of feature attribution for neural networks, a method to identify which parts of an input are most important for the output prediction.
It is based on the *Integrated Gradients* method (http://arxiv.org/abs/1703.01365).
Unlike Integrated Gradients it does not require a choice of *baseline* input, which may resolve some theoretical difficulties.

The project also includes some other related attribution visualisation methods.

**Note:** this is research code. Any aspect may change, particularly overall code structure and interface signatures.

## Usage

Tested with Python version 3.6.9.

Before using this code, install the required modules listed under `install_requires` in the `setup.py` file.

Run the code by executing the `main.py` script in the `src` directory: `python main.py`.

Run the tests by executing `python -m unittest` from inside the `src` directory.

## Example

The image below shows an example of Integrated Certainty Gradients attribution.
A neural network model identifies the input image (left) as "7".
Integrated Certainty Gradients produces the attribution map (center) showing how it made this decision.

![Example attribution](documentation/example%20attribution.png)

This example shows the network was dissuaded by the left part of the cross-arm of the 7 (attributed negatively / in blue).
However, it is reassured strongly by the vacant area around the arm (attributed positively / in orange).
This suggests this part of the image is important for the network in distinguishing it from a "9" (the second most highly scored numeral in this example).

## Correspondence

For communication regarding this project, please contact David Drakard <research@ddrakard.com>.