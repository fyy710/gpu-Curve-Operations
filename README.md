### Curve Operations
This directory contains a reference GPU implementation of 
[Curve Operations](https://codaprotocol.github.io/snark-challenge/problem-04-Curve%20operations.html).


### Build
``` bash
./build.sh
```

### Generate Inputs
``` bash
./generate_inputs
```

### Run
For interpreting inputs in montgomery representation:
``` bash
./main compute inputs outputs
```

### Check results
``` bash
shasum outputs
```
