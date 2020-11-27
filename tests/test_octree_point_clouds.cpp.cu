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

bool allFinite(const float3& v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

class OctreePointCloudTests : public testing::TestWithParam<bool>
{
};

}

TEST_P(OctreePointCloudTests, pc1)
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

        if (allFinite(*v)) {
            src.emplace_back(*v);
        }
    }

    std::ifstream dstFile(dstFilename, std::ios::binary);
    while (dstFile) {
        const auto d1 = readVec(dstFile);
        if (!d1) {
            break;
        }
        const auto d2 = readVec(dstFile);
        CHECK(d2);

        if (allFinite(*d1) && allFinite(*d2)) {
            dst.emplace_back(*d1);
            dstNormals.emplace_back(*d2);
        }
    }

    constexpr size_t N_POINTS = 640*480;
    constexpr size_t N_QUERIES = 63137;

    EXPECT_LE(src.size(), N_POINTS);
    EXPECT_LE(dst.size(), N_POINTS);

    EXPECT_GE(src.size(), N_POINTS * 0.3);
    EXPECT_GE(dst.size(), N_POINTS * 0.3);

    std::mt19937 gen(0);
    std::shuffle(dst.begin(), dst.end(), gen);

    HostDeviceMemory<float3> pointsMem(src);
    HostDeviceMemory<float3> queriesMem(dst);
    HostDeviceMemory<size_t> resultsMem(N_QUERIES);

    // octree_cuda init
    Octree<float3> octree;
    const auto timerInit = Timer();
    if (GetParam()) {
        VLOG(1) << "octree_cuda: CPU initialization";
        octree.initialize(pointsMem.host(), pointsMem.device());
    }
    else {
        VLOG(1) << "octree_cuda: GPU initialization";
        octree.initializeDevice(pointsMem.host(), pointsMem.device());
    }
    VLOG(1) << timerInit.printMs("octree_cuda: init time");

    // validate internal structures
    octree_cuda::OctreeTest t{octree};
    t.check();

    // UniBn Octree init
    unibn::Octree<float3, cudex::HostSpan<float3>> unibn;
    const auto timerInitUnibnInit = Timer();
    const auto hostSpan = pointsMem.host();
    unibn.initialize(hostSpan);
    VLOG(1) << timerInitUnibnInit.printMs("unibn: init time");

    // octree_cuda gpu query
    const auto gpuq = octree.deviceIndex();
    auto launcher = Launcher().sync().size1D(N_QUERIES);

    const auto timerCuda = Timer();
    launcher.run(runQuery<float3>, queriesMem.device().first(N_QUERIES), gpuq, resultsMem.device());
    VLOG(1) << timerCuda.printMs("octree_cuda: cuda time");
    resultsMem.copyDeviceToHost();

    // octree_cuda cpu query
    const auto cpuq = octree.hostIndex();
    const auto timerCPU = Timer();
    const auto cpuResult = runQueryCPU(queriesMem.host().first(N_QUERIES), cpuq);
    VLOG(1) << timerCPU.printMs("octree_cuda: cpu time");

    EXPECT_EQ(cpuResult.size(), N_QUERIES);

    // unibn cpu query
    const auto timerInitUnibn = Timer();
    const auto unibnResult = runQueryCPU<unibn::L2Distance<float3>>(queriesMem.host().first(N_QUERIES), unibn);
    VLOG(1) << timerInitUnibn.printMs("unibn: query time");

    EXPECT_EQ(unibnResult.size(), N_QUERIES);

    // Check results

    for (size_t i=0; i < N_QUERIES; ++i) {
        EXPECT_EQ(cpuResult[i], resultsMem[i]);
        EXPECT_EQ(cpuResult[i], unibnResult[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(OctreePointCloudTests, OctreePointCloudTests, testing::Values(true, false));
