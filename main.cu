#include <string>
#include <chrono>

#define NDEBUG 1

#include <prover_reference_functions.hpp>

#include "multiexp/reduce.cu"

template< typename B >
struct ec_type;

template<>
struct ec_type<mnt4753_libsnark> {
    typedef ECp_MNT4 ECp;
    typedef ECp2_MNT4 ECpe;
};

template<>
struct ec_type<mnt6753_libsnark> {
    typedef ECp_MNT6 ECp;
    typedef ECp3_MNT6 ECpe;
};

int main(int argc, char **argv) {
    mnt4753_libsnark::init_public_params();

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");

    while (true) {
        size_t n;
        size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);

        if (elts_read == 0) { break; }
        printf("n %d\n", n);

        typedef typename ec_type<mnt4753_libsnark>::ECp ECp4;
        typedef typename ec_type<mnt4753_libsnark>::ECpe ECpe4;
        typedef typename mnt4753_libsnark::G1 G1_4;
        typedef typename mnt4753_libsnark::G2 G2_4;

        typedef typename ec_type<mnt6753_libsnark>::ECp ECp6;
        typedef typename ec_type<mnt6753_libsnark>::ECpe ECpe6;
        typedef typename mnt6753_libsnark::G1 G1_6;
        typedef typename mnt6753_libsnark::G2 G2_6;

        auto g4_1 = load_points<ECp4>(n, inputs);
        auto g4_2 = load_points<ECpe4>(n, inputs);
        auto g6_1 = load_points<ECp6>(n, inputs);
        auto g6_2 = load_points<ECpe6>(n, inputs);
#if 0
        printf("g4 1\n");
        uint8_t *buf = (uint8_t *)g6_2.get();
        for (int i = 0; i < 3 * 2 *3 * ELT_BYTES; i ++) {
            printf("%02x",buf[i]); 
        }
        printf("\n");
#endif

        cudaStream_t s4_1, s4_2, s6_1, s6_2;
#if 1
        ec_point_add<ECp4>(s4_1, g4_1.get(), n);
        cudaDeviceSynchronize();
        G1_4 *g1_4 = mnt4753_libsnark::read_pt_ECp(g4_1.get());
        ec_point_add<ECpe4>(s4_2, g4_2.get(), n);
        cudaDeviceSynchronize();
        G2_4 *g2_4 = mnt4753_libsnark::read_pt_ECpe(g4_2.get());
        ec_point_add<ECp6>(s6_1, g6_1.get(), n);
        cudaDeviceSynchronize();
        G1_6 *g1_6 = mnt6753_libsnark::read_pt_ECp(g6_1.get());
#endif
        ec_point_add<ECpe6>(s6_2, g6_2.get(), n);
        cudaDeviceSynchronize();
        G2_6 *g2_6 = mnt6753_libsnark::read_pt_ECpe(g6_2.get());
#if 1
        mnt4753_libsnark::groth16_write(g1_4, g2_4, outputs);
        mnt6753_libsnark::groth16_write(g1_6, g2_6, outputs);


        cudaStreamDestroy(s4_1);
        cudaStreamDestroy(s4_2);
        cudaStreamDestroy(s6_1);
        cudaStreamDestroy(s6_2);
        mnt4753_libsnark::delete_G1(g1_4);
        mnt4753_libsnark::delete_G2(g2_4);
        mnt6753_libsnark::delete_G1(g1_6);
        mnt6753_libsnark::delete_G2(g2_6);
#endif
    }
    return 0;
}
