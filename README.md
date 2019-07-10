### Quadratic Extension
This directory contains a reference GPU implementation of
[Quadratic Extension](https://codaprotocol.github.io/snark-challenge/problem-02-Quadratic%20extension%20arithmetic.html)
using [libff](README-libff.md) and GPU.


### Build
``` bash
./build.sh
```

### Using the inputs generated in CPU

### Run
Support inputs in montgomery representation:
``` bash
./main compute inputs outputs
```

### Check results
``` bash
shasum outputs
```
