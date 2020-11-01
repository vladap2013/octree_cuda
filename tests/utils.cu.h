#pragma once

#include "octree_cuda/octree.cu.h"

#include "cudex/device_utils.cu.h"

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
    ASSERT_EQ(ot.successors_[nPoints() - 1], INVALID_INDEX);

    checkPoints();

    ASSERT_GT(ot.octants_.size(), 0);

    ASSERT_EQ(octant(0).size, nPoints());
    ASSERT_EQ(pointOrdering[octant(0).start], 0);
    ASSERT_EQ(pointOrdering[octant(0).end], nPoints() - 1);

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
    ASSERT_LT(pointOrdering[o.start], pointOrdering[o.end]);
    ASSERT_EQ(o.size, pointOrdering[o.end] - pointOrdering[o.start] + 1);

    if (o.isLeaf) {
        forEachPointInOctant(o, [this, &o](Index pt) {
            const auto p = Point3D(ot.hostPoints_[pt]);
            ASSERT_LE((p - o.center).abs().maxElement(), o.extent);
        });
    }
    else {
        Index end = 0;
        Index beg = static_cast<Index>(-1);

        size_t cnt = 0;

        for(size_t i=0; i < impl::N_OCTANT_CHILDREN; ++i) {
            if (o.children[i] == impl::INVALID_CHILD) {
                continue;
            }

            checkOctant(o.children[i]);

            auto &oc = octant(o.children[i]);
            ASSERT_TRUE(o.containsBall(oc.center, oc.extent));

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
