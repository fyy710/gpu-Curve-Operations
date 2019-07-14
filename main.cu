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
    uint8_t mnt4753_one[96] = {66, 111, 220, 217, 171, 236, 168, 152, 134, 70, 3, 90, 198, 49, 205, 145, 46, 87, 20, 205, 160, 228, 195, 151, 1, 182, 136, 199, 25, 152, 88, 121, 111, 151, 8, 33, 148, 156, 38, 237, 104, 29, 3, 207, 138, 77, 15, 30, 89, 133, 51, 19, 183, 59, 12, 50, 98, 10, 240, 210, 2, 67, 139, 89, 33, 166, 140, 253, 203, 201, 116, 64, 140, 232, 101, 56, 219, 126, 164, 15, 149, 161, 249, 31, 179, 95, 69, 149, 66, 226, 200, 158, 71, 123, 0, 0};
    uint8_t mnt6753_one[96] = {66, 111, 255, 127, 20, 128, 150, 185, 168, 206, 137, 181, 23, 104, 177, 78, 121, 225, 121, 12, 217, 210, 235, 161, 218, 192, 73, 197, 174, 92, 114, 15, 212, 218, 230, 211, 230, 78, 12, 171, 98, 203, 12, 222, 8, 169, 188, 159, 152, 132, 51, 19, 183, 59, 12, 50, 98, 10, 240, 210, 2, 67, 139, 89, 33, 166, 140, 253, 203, 201, 116, 64, 140, 232, 101, 56, 219, 126, 164, 15, 149, 161, 249, 31, 179, 95, 69, 149, 66, 226, 200, 158, 71, 123, 0, 0};

    mnt4753_libsnark::init_public_params();
    mnt6753_libsnark::init_public_params();

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");

    while (true) {
        size_t n;
        size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);

        if (elts_read == 0) { break; }

        typedef typename ec_type<mnt4753_libsnark>::ECp ECp4;
        typedef typename ec_type<mnt4753_libsnark>::ECpe ECpe4;
        typedef typename mnt4753_libsnark::G1 G1_4;
        typedef typename mnt4753_libsnark::G2 G2_4;

        typedef typename ec_type<mnt6753_libsnark>::ECp ECp6;
        typedef typename ec_type<mnt6753_libsnark>::ECpe ECpe6;
        typedef typename mnt6753_libsnark::G1 G1_6;
        typedef typename mnt6753_libsnark::G2 G2_6;

        auto g4_1 = load_points<ECp4>(n, inputs);
        uint8_t *buf;
        buf = (uint8_t *)g4_1.get();
        // add z
        for (int i = 0; i < n; i++) {
           memcpy(buf + (i * 3 + 2) * ELT_BYTES, mnt4753_one, ELT_BYTES);
        }
        auto g4_2 = load_points<ECpe4>(n, inputs);
        buf = (uint8_t *)g4_2.get();
        // add z
        for (int i = 0; i < n; i++) {
           memcpy(buf + (i * 6 + 4) * ELT_BYTES, mnt4753_one, ELT_BYTES);
        }
        auto g6_1 = load_points<ECp6>(n, inputs);
        buf = (uint8_t *)g6_1.get();
        // add z
        for (int i = 0; i < n; i++) {
           memcpy(buf + (i * 3 + 2) * ELT_BYTES, mnt6753_one, ELT_BYTES);
        }
        auto g6_2 = load_points<ECpe6>(n, inputs);
        buf = (uint8_t *)g6_2.get();
        // add z
        for (int i = 0; i < n; i++) {
           memcpy(buf + (i * 9 + 6) * ELT_BYTES, mnt6753_one, ELT_BYTES);
        }

        cudaStream_t s4_1, s4_2, s6_1, s6_2;
        ec_point_add<ECp4>(s4_1, g4_1.get(), n);
        ec_point_add<ECpe4>(s4_2, g4_2.get(), n);
        ec_point_add<ECp6>(s6_1, g6_1.get(), n);
        ec_point_add<ECpe6>(s6_2, g6_2.get(), n);
        cudaDeviceSynchronize();
        G1_4 *g1_4 = mnt4753_libsnark::read_pt_ECp(g4_1.get());
        G2_4 *g2_4 = mnt4753_libsnark::read_pt_ECpe(g4_2.get());
        G1_6 *g1_6 = mnt6753_libsnark::read_pt_ECp(g6_1.get());
        G2_6 *g2_6 = mnt6753_libsnark::read_pt_ECpe(g6_2.get());
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
    }
    return 0;
}
