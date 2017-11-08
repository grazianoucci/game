# GAME benchmarks

> This is just a test file to contain some benchmarks of the main algorithm

## Table of content

- [tl;dr](#tldr)
- [Inputs](#inputs)
- [Setup](#setup)
- [Ouputs](#outputs)

## TL;DR
```
linear trend on the number of features (using 1 model)
```

## Inputs
Inputs have been generated with the [provided script](create_benchmark_inputs.py): for each size, the script samples random features from the [allowed ones](/library/library_labels.dat), and generates random numbers from a uniform distribution (with default `min, max = 1e-9, 1e9`)

## Setup
```bash
python2 create_benchmark_inputs.py
python2 run_benchmarks.py
```
and using just 2 processes under linux (with 8-cores and 16GB RAM) it should take the [following times](#outputs).

## Outputs
A sample of the raw output:
```
→ python2 run_benchmarks.py    
Running test with 10 features
Model 1/1 completed...
Tried to write results to file but got error:
[Errno 20] Not a directory: '/dev/null/model_ids.dat'
Successfully completed test with 10 features
        Time taken: 46.9975318909 seconds

Running test with 72 features
Model 1/1 completed...
Tried to write results to file but got error:
[Errno 20] Not a directory: '/dev/null/model_ids.dat'
Successfully completed test with 72 features
        Time taken: 264.147193193 seconds

Running test with 32 features
Model 1/1 completed...

...

Done all tests, displaying chart...
```

yielding a table like

| Number of features | Time taken |
| --- | --- |
| `2` | `34.3053150177`
| `10` | `46.9975318909`
| `12` | `53.5937390327`
| `22` | `91.2368578911`
| `32` | `129.276766062`
| `42` | `142.909941912`
| `52` | `192.220395088`
| `62` | `227.963886976`
| `72` | `264.147193193`
| `82` | `247.914298058`
| `92` | `321.900949001`

which shows a trend like
![trend plot](extra/benchmarks.png)
