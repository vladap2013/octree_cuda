#include "gtest/gtest.h"
#include "glog/logging.h"

#include "Octree.hpp"

#include "octree_cuda/octree.cu.h"
#include "cudex/launcher.cu.h"

#include "utils.cu.h"

#include <random>
#include <fstream>

using namespace octree_cuda;
using namespace cudex;

namespace {

std::optional<float3> readVec(std::ifstream& f)
{
    EXPECT_EQ(sizeof(float), 4);
    constexpr size_t N_BYTES = 3 * 4;

    float3 data;

    f.read(reinterpret_cast<char*>(&data), N_BYTES);
    const auto cnt = f.gcount();
    EXPECT_TRUE(cnt == 0 || cnt == N_BYTES) << "Read: " << cnt;

    return cnt == N_BYTES ? std::make_optional(data) : std::nullopt;
}

}

TEST(octree_point_cloud, pc1)
{
    using namespace cudex;

    const std::string base = TEST_RESOURCE_DIR;
    const std::string srcFilename = base + "/src.bin";
    const std::string dstFilename = base + "/dst.bin";

    std::vector<float3> src;
    std::vector<float3> dst;
    std::vector<float3> dstNormals;

    // Read data

    std::ifstream srcFile(srcFilename, std::ios::binary);
    CHECK(srcFile) << srcFilename;

    while (srcFile) {
        const auto v = readVec(srcFile);

        if (!v) {
            break;
        }

        src.emplace_back(*v);
    }

    std::ifstream dstFile(dstFilename, std::ios::binary);
    while (dstFile) {
        const auto d1 = readVec(dstFile);
        if (!d1) {
            break;
        }
        const auto d2 = readVec(dstFile);
        CHECK(d2);

        dst.emplace_back(*d1);
        dstNormals.emplace_back(*d2);
    }

    constexpr size_t N_POINTS = 640*480;

    EXPECT_EQ(src.size(), N_POINTS);
    EXPECT_EQ(dst.size(), N_POINTS);
    EXPECT_EQ(dstNormals.size(), N_POINTS);
}
