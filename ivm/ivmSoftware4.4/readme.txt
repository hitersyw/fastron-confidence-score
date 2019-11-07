Matlab implementation of a revised version of Import Vector Machine classifier (Zhu and Hastie 2005)


Table of Contents
=================

- Introduction
- Installation
- Usage
- Determining the kernel and regularization parameter

Introduction
============

This implementation is based on the Import Vector Machine algorithm of Zhu and Hastie.
The algorithm is a sparse, probabilistic and discriminative Kernel Logistic Regression model.
It shows similar accuracy like the Support Vector Machines, but has a probabilistic output.
The model is also sparser, so that the classification step is faster.

Installation
============

On Unix systems, we recommend using GNU g++ as your compiler.

On all systems just navigate into the folder src and type 'make' into your Matlab command 
window to build the mex-files.

The implementation uses the Eigen library (http://eigen.tuxfamily.org/dox/).

Usage
=====

params = init;
    
    - initialization of all parameters

result = ivm(data, params);

A toy example is given with the function main_IVMdna.m with and main_IVMripley.

Determining the kernel and regularization parameter
===================================================

The determination of the kernel parameter is done via gridsearch and crossvalidation.

Greedy selection of import vectors
==================================
You can optionally use a hybrid forward/backward strategy, which successively add import vectors to the set
(forward step), but also test in each step if import vectors can be removed (backward step). 
Since we start with an empty import vector set, and only add import vectors sequentially, 
in the first iterations the decision boundary can be very different from their final position.
Therefore a removal of import points can lead to a sparser and more accurate solution than 
only using forward selection steps. On the other side this step is time consuming.
     
Publication of the Import Vector Machine:
=========================================

Zhu, Ji /  Hastie, Trevor: 
Kernel Logistic Regression and the Import Vector Machine 
In: Journal of Computational and Graphical Statistics 14, 1, 2005, pp. 185-205. 
http://pubs.amstat.org/doi/abs/10.1198/106186005X25619.

Roscher, Ribana / Förstner, Wolfgang / Waske, Björn:
I^2VM: Incremental Import Vector Machines
In: Image and Vision Computing 30, 4-5, 2012, pp. 263-278.
(Please cite this paper if you use the software)
