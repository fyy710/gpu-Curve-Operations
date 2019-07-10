#include <string>
#include <chrono>

#define NDEBUG 1

#include <prover_reference_functions.hpp>

#include "multiexp/reduce.cu"

int main(int argc, char **argv) {
    mnt4753_libsnark::init_public_params();

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");

    while (true) {
        size_t n;
        size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);

        if (elts_read == 0) { break; }

        size_t data_size = n * 2 * ELT_BYTES;

        auto x = allocate_memory(data_size);
        fread((void *)x.get(), data_size, 1, inputs);
        auto y = allocate_memory(data_size);
        fread((void *)y.get(), data_size, 1, inputs);
        auto out = allocate_memory(data_size);

        ec_multi<Fp2_MNT4>(x.get(), y.get(), out.get(), n);
        cudaDeviceSynchronize();
        fwrite((void *)out.get(), data_size, 1, outputs);
    }
    return 0;
}
