# crc-hamming-distance

Independent verification of Hamming weights for various CRC polynomials
using CUDA.

[Koopman et al.](http://users.ece.cmu.edu/~koopman/roses/dsn04/koopman04_crc_poly_embedded.pdf) 
define a Hamming weight with respect to a CRC polynomial as being a
function of three inputs:

* A CRC polynomial **p**
* A message size **m** in bits
* Some number of error bit flips **k**

Formally, a Hamming weight is the number of all possible **k**-bit 
errors in a message of length **m** that are undetected by the
CRC polynomial **p**. A Hamming weight of 0 would indicate
that a CRC polynomial is able to detect every **k**-bit
error combination in a message of size **m**.

![Hamming Weight Table](https://i.stack.imgur.com/voOrN.png)

A Hamming distance of a CRC polynomial is defined as a stronger
metric that incorporates Hamming weights; a polynomial with
Hamming distance **hd** at a certain message length has Hamming
weight 0 for all **k**-bit error bit combinations where
**k < hd**. This is often used as a criteria for judging the
effectiveness of a CRC polynomial.

Calculating a Hamming weight unfortunately takes a large
amount of computational resources, as every **k**-bit error
combination has to be checked. The number of combinations
per Hamming weight grows roughly exponentially with respect
to **k**.

That being said, all hope is not lost. Actually checking 
every combination is an easy parallelizable task and
the CRC validation is inexpensive. Therefore, it would
make sense that a GPU would be calculate Hamming weights
quickly, owing to its large number of threads. This project
is an attempt to implement calculation of Hamming
weights using NVIDIA GPUs via CUDA.

## Usage

Perform an out-of-source CMake build. You must have CUDA
installed. CUDA itself will only work with NVIDIA GPUs.

```
mkdir build
cd build
cmake ..
make -j8
```

# Restrictions

There are some restrictions to the weights calculation program.

1. CRC polynomials must be provided in implicit +1
(e.g. Koopman) notation. That is, the polynomial assumes
that the least significant term is always a 1.

2. Only CRC polynomials less than or equal to 64 bits are accepted.

3. The maximum number of combinations that the GPU evaluates must
fit within a 64-bit unsigned integer. That is:

```
((m + length(p)) choose k < 2^64
```

This is simply because combination indices are calculated using
64-bit integers. In the future, I would like to move to 128-bit
integers which would bump up this limit even more.
