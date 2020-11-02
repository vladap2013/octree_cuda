#include "gtest/gtest.h"
#include "glog/logging.h"

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

}

TEST(octree_cuda, cube)
{
    const std::vector<TestPoint> points = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
    };

    const std::vector<TestPoint> queries = {
        { 0, 0, 0 },
        { 0.4, 0.3, 0.2 },
        { -4, -4, 2 },
        { 10, 0, 0 },
        { 5, 5, 5 }
    };

    const size_t N = queries.size();

    const std::vector<size_t> results = { 0, 0, 4, 1, 6 };
    EXPECT_EQ(N, results.size());

    HostDeviceMemory<TestPoint> pointsMem(points);
    HostDeviceMemory<TestPoint> queriesMem(queries);

    Octree<TestPoint> octree;
    octree.initialize(pointsMem.host(), pointsMem.device());

    const auto deviceq = octree.deviceIndex();
    const auto hostq = octree.hostIndex();

    HostDeviceMemory<size_t> indexMem(N);

    auto launcher = Launcher().sync().size1D(N);
    launcher.run(runQuery<TestPoint>, queriesMem.device(), deviceq, indexMem.device());

    indexMem.copyDeviceToHost();

    for (size_t i=0; i < N; ++i) {
        EXPECT_EQ(indexMem[i], results[i]);

        EXPECT_EQ(hostq.findNeighbor(queries[i]), results[i]);
    }

    OctreeTest t{octree};
    t.check();
}

TEST(octree_cuda, random)
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
    octree.initialize(pointsMem.host(), pointsMem.device());
    VLOG(1) << timerInit.printMs("octree_cuda: init time");

    // validate internal structures
    OctreeTest t{octree};
    t.check();

    // octree_cuda gpu query
    const auto gpuq = octree.deviceIndex();
    auto launcher = Launcher().sync().size1D(N_QUERIES);

    const auto timerCuda = Timer();
    launcher.run(runQuery<float3>, queriesMem.device(), gpuq, resultsMem.device());
    VLOG(1) << timerCuda.printMs("octree_cuda: cuda time");

    // octree_cuda cpu query
    resultsMem.copyDeviceToHost();
    const auto cpuq = octree.hostIndex();
    const auto timerCPU = Timer();
    auto cpuResult = runQueryCPU(queriesMem.host(), cpuq);
    VLOG(1) << timerCPU.printMs("octree_cuda: cpu time");

    EXPECT_EQ(cpuResult.size(), queriesMem.size());

    // const auto timerCPU = Timer();
    // for (size_t i=0; i < N_QUERIES; ++i) {
    //     const size_t minTree = octree.findNeighbor(queries[i]);
    //     EXPECT_EQ(resultsMem[i], minTree);
    // }
    // VLOG(1) << timerCPU.printMs("CPU timer");

    for (size_t i=0; i < 50; ++i) {
        const size_t minLoop = findClosest(queriesMem[i], pointsMem.host());
        EXPECT_EQ(cpuResult[i], minLoop) << "cpu: " << i;
        EXPECT_EQ(resultsMem[i], minLoop) << "gpu: " << i;
    }
}


