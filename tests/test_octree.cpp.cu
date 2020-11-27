#include "gtest/gtest.h"
#include "glog/logging.h"

#include "Octree.hpp"

#include "octree_cuda/octree.cu.h"
#include "cudex/launcher.cu.h"

#include "utils.cu.h"

#include <random>

using namespace octree_cuda;
using namespace cudex;

namespace {

struct TestPoint
{
    float xval, yval, zval;
};

template<size_t N>
__host__ __device__ float get(const TestPoint& point)
{
    if constexpr(N == 0) return point.xval;
    if constexpr(N == 1) return point.yval;

    return point.zval;
}


class OctreeBasicTests : public testing::TestWithParam<bool>
{
};

}



// TEST_P(OctreeBasicTests, cube)
// {
//     const std::vector<TestPoint> points = {
//         {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
//         {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
//     };
//
//     const std::vector<TestPoint> queries = {
//         { 0, 0, 0 },
//         { 0.4, 0.3, 0.2 },
//         { -4, -4, 2 },
//         { 10, 0, 0 },
//         { 5, 5, 5 }
//     };
//
//     const size_t N = queries.size();
//
//     const std::vector<size_t> results = { 0, 0, 4, 1, 6 };
//     EXPECT_EQ(N, results.size());
//
//     HostDeviceMemory<TestPoint> pointsMem(points);
//     HostDeviceMemory<TestPoint> queriesMem(queries);
//
//     Octree<TestPoint> octree;
//     if (GetParam()) {
//         octree.initialize(pointsMem.host(), pointsMem.device());
//     }
//     else {
//         octree.initializeDevice(pointsMem.host(), pointsMem.device());
//     }
//
//     const auto deviceq = octree.deviceIndex();
//     const auto hostq = octree.hostIndex();
//
//     HostDeviceMemory<size_t> indexMem(N);
//
//     auto launcher = Launcher().sync().size1D(N);
//     launcher.run(runQuery<TestPoint>, queriesMem.device(), deviceq, indexMem.device());
//
//     indexMem.copyDeviceToHost();
//
//     for (size_t i=0; i < N; ++i) {
//         EXPECT_EQ(indexMem[i], results[i]);
//
//         EXPECT_EQ(hostq.findNeighbor(queries[i]), results[i]);
//     }
//
//     octree_cuda::OctreeTest t{octree};
//     t.check();
// }

TEST_P(OctreeBasicTests, random)
{
    constexpr size_t N_POINTS = 200000;
    constexpr size_t N_QUERIES = 5331;

    constexpr float POINT_RANGE = 100;
    constexpr float QUERY_RANGE = 120;

    std::uniform_real_distribution<float> pointDist(-POINT_RANGE, POINT_RANGE);
    std::uniform_real_distribution<float> queryDist(-QUERY_RANGE, QUERY_RANGE);
    std::mt19937 gen(0);

    HostDeviceMemory<float3> pointsMem;
    HostDeviceMemory<float3> queriesMem;

    pointsMem.resizeSync(genRandomVector(N_POINTS, pointDist, gen));
    queriesMem.resizeSync(genRandomVector(N_QUERIES, queryDist, gen));

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
    launcher.run(runQuery<float3>, queriesMem.device(), gpuq, resultsMem.device());
    VLOG(1) << timerCuda.printMs("octree_cuda: cuda time");
    resultsMem.copyDeviceToHost();

    // octree_cuda cpu query
    const auto cpuq = octree.hostIndex();
    const auto timerCPU = Timer();
    const auto cpuResult = runQueryCPU(queriesMem.host(), cpuq);
    VLOG(1) << timerCPU.printMs("octree_cuda: cpu time");

    EXPECT_EQ(cpuResult.size(), queriesMem.size());

    // unibn cpu query
    const auto timerInitUnibn = Timer();
    const auto unibnResult = runQueryCPU<unibn::L2Distance<float3>>(queriesMem.host(), unibn);
    VLOG(1) << timerInitUnibn.printMs("unibn: query time");

    EXPECT_EQ(unibnResult.size(), queriesMem.size());

    for (size_t i=0; i < N_QUERIES; ++i) {
        EXPECT_EQ(cpuResult[i], resultsMem[i]);
        EXPECT_EQ(cpuResult[i], unibnResult[i]);
    }

    for (size_t i=0; i < 50; ++i) {
        const size_t minLoop = findClosest(queriesMem[i], pointsMem.host());
        EXPECT_EQ(cpuResult[i], minLoop) << "cpu: " << i;
        EXPECT_EQ(resultsMem[i], minLoop) << "gpu: " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(OctreeBasicTests, OctreeBasicTests, testing::Values(true, false));

