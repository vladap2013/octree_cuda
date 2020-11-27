#pragma once

#include "octree_cuda/octree.cu.h"

#include "cudex/device_utils.cu.h"

#include <string>
#include <chrono>
#include <random>
#include <type_traits>

template<typename Point>
__global__ void runQuery(cudex::DeviceSpan<Point> queries,
        octree_cuda::DeviceOctreeIndex<Point> q,
        cudex::DeviceSpan<size_t> result)
{
    const auto index = cudex::threadLinearIndex();
    if (index >= queries.size()) {
        return;
    }

    const size_t r = q.findNeighbor(queries[index]);
    result[index] = r;
}

template<typename P=void, typename Point, typename T>
std::vector<size_t> runQueryCPU(cudex::HostSpan<Point> queries, const T& q)
{
    std::vector<size_t> result(queries.size());
    for (size_t i=0; i < queries.size(); ++i) {
        if constexpr(std::is_same_v<P, void>) {
            result[i] = q.findNeighbor(queries[i]);
        }
        else {
            result[i] = q.template findNeighbor<P>(queries[i]);
        }
    }

    return result;
}


template<typename Dist, typename Gen>
std::vector<float3> genRandomVector(size_t n, Dist& dist, Gen& gen)
{
    std::vector<float3> ret(n);
    for (auto& v: ret) {
        v.x = dist(gen);
        v.y = dist(gen);
        v.z = dist(gen);
    }

    return ret;
}

inline size_t findClosest(const float3& query, cudex::HostSpan<const float3> points)
{
    auto dist2 = [](float3 p1, float3 p2) {
        const float x = p1.x - p2.x;
        const float y = p1.y - p2.y;
        const float z = p1.z - p2.z;

        return x*x + y*y + z*z;
    };

    auto m = std::min_element(points.begin(), points.end(),
        [&](const float3& v1, const float3& v2) {
            return dist2(query, v1) < dist2(query, v2);
        }
    );

    return m - points.begin();
}


// -------------------------------------------------------------------------------------------------
// OctreeTest class

namespace octree_cuda {

template<typename Point>
class OctreeTest
{
public:
    OctreeTest(const Octree<Point>& octree) : ot(octree)
    {}

    void check();

    size_t nPoints() const
    {
        return ot.hostPoints_.size();
    }

    const impl::Octant& octant(Index idx) const
    {
        return ot.octants_[idx];
    }

private:
    void checkPoints();
    void checkOctant(Index);

public:
    const Octree<Point>& ot;
};

template<typename Point>
void OctreeTest<Point>::check()
{
    ASSERT_GT(ot.hostPoints_.size(), 0);
    ASSERT_EQ(ot.pointIndexes_.size(), nPoints());

    checkPoints();

    ASSERT_GT(ot.octants_.size(), 0);

    ASSERT_EQ(octant(0).count, nPoints());
    ASSERT_EQ(octant(0).start, 0);

    checkOctant(0);
}

template<typename Point>
void OctreeTest<Point>::checkPoints()
{
    auto seen = std::vector<bool>(nPoints(), false);

    for (size_t i = 0; i < nPoints(); ++i) {
        const Index ptIndex = ot.pointIndexes_[i];

        ASSERT_NE(ptIndex, INVALID_INDEX);
        ASSERT_LT(ptIndex, ot.hostPoints_.size());

        ASSERT_FALSE(seen[ptIndex]);
        seen[ptIndex] = true;
    }
}

template<typename Point>
void OctreeTest<Point>::checkOctant(Index idx)
{
    auto& o = octant(idx);

    ASSERT_GT(o.count, 0);

    if (o.isLeaf) {
        for (size_t i = 0; i < o.count; ++i) {
            const Index ptIndex = ot.pointIndexes_[o.start + i];
            const auto p = Point3D(ot.hostPoints_[ptIndex]);
            ASSERT_LE((p - o.center).abs().maxElement() * 0.99999, o.extent)
                << "Octant: " << idx << ", ptindex: " << ptIndex << ", local index: " << i;
        };
    }
    else {
        size_t cnt = 0;

        Index prevEnd = INVALID_INDEX;

        for(size_t i=0; i < impl::N_OCTANT_CHILDREN; ++i) {
            if (o.children[i] == INVALID_INDEX) {
                continue;
            }

            checkOctant(o.children[i]);

            auto &oc = octant(o.children[i]);

            // oc.extent is sligthly decreased because of numerical errors.
            ASSERT_TRUE(
                o.containsBall(oc.center, oc.extent * 0.99999)
            ) << o.center << "; " << o.extent << " - " << oc.center << "; " << oc.extent;

            ASSERT_EQ(oc.start, prevEnd != INVALID_INDEX ? prevEnd + 1: o.start);

            prevEnd = oc.start + oc.count - 1;
            cnt += oc.count;
        }

        ASSERT_EQ(cnt, o.count);
        ASSERT_EQ(prevEnd, o.start + o.count - 1);
    }
}

}

// -------------------------------------------------------------------------------------------------
// Timer class

class Timer
{
public:
    Timer() : start_(std::chrono::high_resolution_clock::now())
    {}

    double elapsedMs() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_
        ).count() / 1000.;
    }

    std::string printMs(const std::string& message) const {
        return message + ": " + std::to_string(elapsedMs()) + " ms";
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

