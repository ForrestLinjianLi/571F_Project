## Environment Requirement

The code runs well under python 3.7.7. The required packages are as follows:

- Tensorflow-gpu == 1.15.0
- numpy == 1.19.1
- scipy == 1.5.2
- pandas == 1.1.1
- cython == 0.29.21

## Quick Start
**Firstly**, compline the evaluator of cpp implementation with the following command line:

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

Further details, please refer to [NeuRec](https://github.com/wubinzzu/NeuRec/)

**Secondly**, specify dataset and recommender in configuration file *NeuRec.properties*.

Model specific hyperparameters are in configuration file *./conf/KLGCN.properties*.


**Finally**, run [main.py](./main.py) in IDE or with command line:

```bash
python main.py --recommender=KLGCN
```

Some important parameters:

### book_crossing (book)
lr = 0.0005
reg = 1e-4
n_layers = 3
n_iters = 4
n_neighbors = 2

### last-fm (music)
lr = 0.001
reg = 1e-3
n_layers = 3
n_iters = 1
n_neighbors = 8

### MovieLens-20M (movie)
lr = 0.001
reg = 1e-5
n_layers = 3
n_iters = 1
n_neighbors = 8
