#include "gtest/gtest.h"

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


