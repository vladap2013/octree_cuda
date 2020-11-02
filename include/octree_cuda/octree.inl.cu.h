#include "cudex/uarray.cu.h"
#include "cudex/stack.cu.h"

// -------------------------------------------------------------------------------------------------
// Impl

namespace octree_cuda::impl {

using OctantIndex = Index;

constexpr inline size_t MAX_STACK_SIZE = 32;

constexpr inline uint8_t INVALID_CHILD = 255;
constexpr inline uint8_t N_OCTANT_CHILDREN = 8;

struct Octant
{
    Point3D center;
    float extent;

    bool isLeaf;

    Index start;
    Index end;
    size_t size;

    cudex::UArray<Index, N_OCTANT_CHILDREN> children;

    // Checks if the open ball belongs to the quadrant
    __host__ __device__ bool containsBall(const Point3D& ballCenter, const float radius) const
    {
        assert(isfinite(radius));
        assert(isfinite(extent));

        // We use the assumption that norm(x) >= abs(x_i), for all i,
        // and that octant is a box with axis aligned sides.
        const Point3D diff = (center - ballCenter).abs();
        return (diff + radius).maxElement() < extent;
    }

    // Checks if the open ball intersects the quadrant
    __host__ __device__ bool overlapsBall(const Point3D& ballCenter, const float radius) const
    {
        if (isinf(radius)) {
            return true;
        }

        assert(isfinite(radius));
        assert(isfinite(extent));

        const Point3D diff = (center - ballCenter).abs();

        if (diff.maxElement() >= radius + extent) {
            // If distance in any coordinate between centers is >= than maxDist, there is no intersection
            // We use the assumption that norm(x) >= abs(x_i), for all i.
            return false;
        }

        int nLessExtent = 0;
        const Point3D diff2 = (diff - extent).max(Point3D(0, 0, 0));

        for (auto d: {0, 1, 2}) {
            if (diff[d] <= extent) {
                ++nLessExtent;
            }
        }

        if (nLessExtent >= 2) {
            // ball center belongs to a projection of a quadrant on one of coordinate planes, and
            // there is an overlap in the third coordinate (which is checked in the previous step).

            return true;
        }

        return diff2.squaredNormL2() < radius * radius;
    }
};

struct StackState
{
    Index octantIndex = INVALID_INDEX;
    uint8_t bestChild = INVALID_CHILD;
    uint8_t currentChild = INVALID_CHILD;
};


} // namespace octree_cuda::impl


namespace octree_cuda {

// -------------------------------------------------------------------------------------------------
// Octree

template<typename Point>
Octree<Point>::Octree(const Params& params) : params_(params)
{}

template<typename Point>
Octree<Point>::~Octree() = default;

template<typename Point>
auto Octree<Point>::hostIndex() const -> HostOctreeIndex<Point>
{
    assert(!hostPoints_.empty());
    return {hostPoints_, cudex::makeSpan(octants_), cudex::makeSpan(successors_)};
}

template<typename Point>
auto Octree<Point>::deviceIndex() -> DeviceOctreeIndex<Point>
{
    if (devicePoints_.empty()) {
        throw std::runtime_error("Device points not given");
    }

    octantsMem_.resizeCopy(cudex::makeSpan(octants_));
    successorsMem_.resizeCopy(cudex::makeSpan(successors_));

    return {devicePoints_, octantsMem_.span(), successorsMem_.span()};
}

template<typename Point>
void Octree<Point>::initialize(cudex::HostSpan<const Point> hostPoints)
{
    hostPoints_ = hostPoints;
    devicePoints_ = cudex::DeviceSpan<const Point>();

    if (hostPoints.empty()) {
        throw std::runtime_error("No points");
    }
    makeOctantTree();
}

template<typename Point>
void Octree<Point>::initialize(cudex::HostSpan<const Point> hostPoints, cudex::DeviceSpan<const Point> devicePoints)
{
    if (hostPoints.empty()) {
        throw std::runtime_error("No points");
    }
    if (hostPoints.size() != devicePoints.size()) {
        throw std::runtime_error("Different sizes");
    }

    hostPoints_ = hostPoints;
    devicePoints_ = devicePoints;

    makeOctantTree();
}

template<typename Point>
void Octree<Point>::makeOctantTree()
{
    octants_.clear();
    Point3D mean(0, 0, 0);

    const size_t nPoints = hostPoints_.size();

    successors_.resize(hostPoints_.size());

    Point3D minValues = Point3D(hostPoints_[0]);
    Point3D maxValues = minValues;

    for (size_t i=0; i < nPoints; ++i) {
        const Point3D p = Point3D(hostPoints_[i]);

        minValues = minValues.min(p);
        maxValues = maxValues.max(p);

        successors_[i] = (i < nPoints - 1)? i + 1 : INVALID_INDEX;
    }

    const Point3D center = (minValues + maxValues) / 2;
    const Point3D extents = maxValues - minValues;

    assert(extents.minElement() >= 0);

    makeOctant(center, extents.maxElement(), 0, nPoints - 1, nPoints, 0);
}

template<typename Point>
Index Octree<Point>::makeOctant(
        const Point3D& center,
        const float extent,
        const Index startPoint,
        const Index endPoint,
        const size_t size,
        const size_t level)
{
    if (level == impl::MAX_STACK_SIZE) {
        throw std::runtime_error("Cannot build octree: too many levels");
    }

    const Index octantIdx = octants_.size();
    octants_.emplace_back();

    assert(startPoint != INVALID_INDEX);
    assert(endPoint != INVALID_INDEX);

    impl::Octant o;
    o.center = center;
    o.extent = extent;
    o.size = size;

    o.isLeaf = size <= params_.bucketSize || extent <= 2 * params_.minExtent;

    for (auto& c: o.children) {
        c = INVALID_INDEX;
    }

    if (o.isLeaf) {
        o.start = startPoint;
        o.end = endPoint;

        octants_[octantIdx] = o;
        return octantIdx;
    }

    struct ChildInfo
    {
        Index start = INVALID_INDEX;
        Index end = INVALID_INDEX;
        size_t count = 0;
    };

    std::array<ChildInfo, impl::N_OCTANT_CHILDREN> childInfo;

    Index pointIdx = startPoint;
    for (size_t cnt = 0; cnt < size; ++cnt)
    {
        assert(pointIdx != INVALID_INDEX);

        const Point3D p = Point3D(hostPoints_[pointIdx]);
        const auto childIdx = center.mortonCode(p);
        assert(childIdx < impl::N_OCTANT_CHILDREN);

        auto& info = childInfo[childIdx];

        info.count ++;

        if (info.end == INVALID_INDEX) {
            info.end = pointIdx;
        }

        const Index nextIdx = successors_[pointIdx];

        successors_[pointIdx] = info.start;
        info.start = pointIdx;

        assert(cnt < size - 1 || pointIdx == endPoint);
        pointIdx = nextIdx;
    }

    Index lastChildIdx = INVALID_INDEX;
    for (size_t childIdx = 0; childIdx < impl::N_OCTANT_CHILDREN; ++childIdx) {
        const auto& info = childInfo[childIdx];

        if (info.count == 0) {
            continue;
        }

        assert(info.start != INVALID_INDEX);
        assert(info.end != INVALID_INDEX);
        assert(successors_[info.end] == INVALID_INDEX);

        const float oExtent = extent * 0.5;

        Point3D move;
        for (int d : {0, 1, 2}) {
            move[d] = ((childIdx & (1 << d)) > 0 ? 1 : -1) * oExtent;
        }

        const Point3D oCenter = center + move;

        Index& child = o.children[childIdx];
        child = makeOctant(oCenter, oExtent, info.start, info.end, info.count, level + 1);

        assert(child != INVALID_INDEX);
        assert(octants_[child].size == info.count);

        if (lastChildIdx == INVALID_INDEX) {
            o.start = octants_[child].start;
        }
        else {
            impl::Octant& lastChildOctant = octants_[o.children[lastChildIdx]];

            assert(lastChildIdx < childIdx);
            assert(successors_[lastChildOctant.end] == INVALID_INDEX);

            successors_[lastChildOctant.end] = octants_[child].start;
        }

        lastChildIdx = childIdx;
    }

    assert(lastChildIdx != impl::INVALID_CHILD);
    o.end = octants_[o.children[lastChildIdx]].end;

    assert(successors_[o.end] == INVALID_INDEX);

    octants_[octantIdx] = o;
    return octantIdx;
}


// -------------------------------------------------------------------------------------------------
// OctreeIndex

template<bool isDevice, typename Point>
__host__ OctreeIndex<isDevice, Point>::OctreeIndex(
        Span<const Point> points,
        Span<const impl::Octant> octants,
        Span<const Index> successors)
    : points_(points)
    , octants_(octants)
    , successors_(successors)
{}

template<bool isDevice, typename Point>
__host__ __device__ size_t OctreeIndex<isDevice, Point>::nPoints() const
{
    return points_.size();
}

template<bool isDevice, typename Point>
__host__ __device__ const Point& OctreeIndex<isDevice, Point>::point(Index index) const
{
    return points_[index];
}

template<bool isDevice, typename Point>
__host__ __device__ Index OctreeIndex<isDevice, Point>::findNeighbor(const Point& query, const float minDistance) const
{
    return findNeighbor(Point3D(query), minDistance);
}

template<bool isDevice, typename Point>
__host__ __device__ Index OctreeIndex<isDevice, Point>::findNeighbor(const Point3D& query, const float minDistance) const
{
#ifdef __CUDA_ARCH__
    assert(isDevice);
#else
    assert(!isDevice);
#endif


    float maxDistance = INFINITY;
    float maxDistance2 = INFINITY;

    const float minDistance2 = minDistance < 0 ? minDistance : minDistance * minDistance;

    using StackState = impl::StackState;

    cudex::Stack<StackState, impl::MAX_STACK_SIZE> stack;
    stack.push(StackState{0});

    Index closest = INVALID_INDEX;

    bool stop = false;
    while(!stack.empty() && !stop)
    {
        StackState& state = stack.top();

        assert(state.octantIndex != INVALID_INDEX);
        assert(state.bestChild < 8 || state.bestChild == impl::INVALID_CHILD);

        if (state.currentChild == 8) {
            stack.pop();
            continue;
        }

        const Octant& octant = octants_[state.octantIndex];

        // Check leaf
        if (octant.isLeaf)
        {
            assert(octant.size > 0);

            Index currentIndex = octant.start;
            for (size_t i = 0; i < octant.size; ++i)
            {
                assert(currentIndex != INVALID_INDEX);

                const float dist2 = (query - Point3D(points_[currentIndex])).squaredNormL2();
                if (minDistance2 < dist2 && dist2 < maxDistance2)
                {
                    maxDistance2 = dist2;
                    closest = currentIndex;
                }

                currentIndex = successors_[currentIndex];
            }

            maxDistance = sqrt(maxDistance2);
            stop = octant.containsBall(query, maxDistance);

            stack.pop();
            continue;
        }

        // Find most probable child
        if (state.currentChild == impl::INVALID_CHILD) {
            assert(state.bestChild == impl::INVALID_CHILD);

            state.currentChild = 0;
            uint8_t bestChild = octant.center.mortonCode(query);

            const Index childIndex = octant.children[bestChild];

            if (childIndex != INVALID_INDEX) {
                state.bestChild = bestChild;
                stack.push(StackState{childIndex});
                continue;
            }
        }

        assert(state.currentChild < 8);

        for (bool childPushed = false; state.currentChild < 8 && !childPushed; ++state.currentChild) {
            if (state.currentChild == state.bestChild) {
                continue;
            }

            const Index childIndex = octant.children[state.currentChild];
            if (childIndex == INVALID_INDEX) {
                continue;
            }

            const Octant& childOctant = octants_[childIndex];
            if (!childOctant.overlapsBall(query, maxDistance)) {
                continue;
            }

            stack.push(StackState{childIndex});
            childPushed = true;
        }
    }

    return closest;
}


}
