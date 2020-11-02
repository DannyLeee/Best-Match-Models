## Introduction
* This is the Information Retrieval HW2
* Using TF-IDF to compute the relation between given querys and documents

## Usage
```
code.py [-h] [-B B] [-K1 K1] [-K3 K3] [-use_q_tf {T,F}]

optional arguments:
  -h, --help       show this help message and exit
  -B B             (default: 0.75)
  -K1 K1           (default: 3.5)
  -K3 K3           (default: 1000)
  -use_q_tf {T,F}  whether to use query's tf; will deactivate K3 (default: F)
```

## Approach
* IDF different from original formula
    * Square IDF make it more important
    * <img src="https://latex.codecogs.com/gif.latex?IDF = (ln(1 + \frac{N+0.5}{n_i+0.5}))^2"/>
* <img src="https://latex.codecogs.com/gif.latex?b = 0.75"/>
* <img src="https://latex.codecogs.com/gif.latex?K_1 = 3.5"/>
* not use query's TF term that in formula (having <img src="https://latex.codecogs.com/gif.latex?K_3"/>'s term), i.e.
    * <img src="https://latex.codecogs.com/gif.latex?\frac{(K_3+1) \times tf_{i,q}}{K_3 \times tf_{i,q}} = 1"/>