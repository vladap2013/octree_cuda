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

    template<typename F>
    void forEachPointInOctant(const impl::Octant& octant, F f);

private:
    void checkPoints();
    void checkOctant(Index);

public:
    const Octree<Point>& ot;
    std::vector<Index> pointOrdering;
};

template<typename Point>
template<typename F>
void OctreeTest<Point>::forEachPointInOctant(const impl::Octant& o, F f)
{
    Index pt = o.start;
    Index ptPrev = INVALID_INDEX;

    for (size_t i=0; i < o.size; ++i, pt = ot.successors_[pt]) {
        ASSERT_TRUE(pt != o.end || i == o.size-1);
        ASSERT_NE(pt, INVALID_INDEX);
        ASSERT_LT(pt, nPoints());

        ptPrev = pt;
        f(pt);
    }

    CHECK_EQ(ptPrev, o.end);
}

template<typename Point>
void OctreeTest<Point>::check()
{
    ASSERT_GT(ot.hostPoints_.size(), 0);

    ASSERT_EQ(ot.successors_.size(), nPoints());

    checkPoints();

    ASSERT_GT(ot.octants_.size(), 0);

    ASSERT_EQ(octant(0).size, nPoints());
    ASSERT_EQ(pointOrdering[octant(0).start], 0);
    ASSERT_EQ(pointOrdering[octant(0).end], nPoints() - 1);
    ASSERT_EQ(ot.successors_[octant(0).end], INVALID_INDEX);

    checkOctant(0);
}

template<typename Point>
void OctreeTest<Point>::checkPoints()
{
    pointOrdering = std::vector<Index>(nPoints(), INVALID_INDEX);

    size_t cnt = 0;
    forEachPointInOctant(octant(0), [this, &cnt](Index pt) {
        ASSERT_EQ(pointOrdering[pt], INVALID_INDEX);
        pointOrdering[pt] = cnt++;
    });

    ASSERT_EQ(cnt, nPoints());
}

template<typename Point>
void OctreeTest<Point>::checkOctant(Index idx)
{

    auto& o = octant(idx);

    ASSERT_GT(o.size, 0);
    ASSERT_LE(pointOrdering[o.start], pointOrdering[o.end]);
    ASSERT_EQ(o.size, pointOrdering[o.end] - pointOrdering[o.start] + 1);

    if (o.isLeaf) {
        forEachPointInOctant(o, [this, &o](Index pt) {
            const auto p = Point3D(ot.hostPoints_[pt]);
            ASSERT_LE((p - o.center).abs().maxElement() * 0.99999, o.extent);
        });
    }
    else {
        Index end = 0;
        Index beg = static_cast<Index>(-1);

        size_t cnt = 0;

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

            beg = std::min(beg, pointOrdering[oc.start]);
            end = std::max(end, pointOrdering[oc.end]);
            cnt += oc.size;
        }

        ASSERT_EQ(cnt, o.size);
        ASSERT_EQ(beg, pointOrdering[o.start]);
        ASSERT_EQ(end, pointOrdering[o.end]);
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

