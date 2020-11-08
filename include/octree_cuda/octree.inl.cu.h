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

    cudex::UArray<Index, N_OCTANT_CHILDREN> children;

    __host__ __device__ size_t size() const
    {
        assert(start <= end);
        return end - start + 1;
    }

    // Checks if the open ball belongs to the quadrant
    __host__ __device__ bool containsBall(const Point3D& ballCenter, const float radius) const
    {
        assert(isfinite(radius));
        assert(isfinite(extent));

        const Point3D diff = (center - ballCenter).abs();
        return (diff + radius).maxElement() <= extent;
    }

    // Checks if the open ball intersects the quadrant
    __host__ __device__ bool overlapsBall(const Point3D& ballCenter, const float radius) const
    {
        if (isinf(radius)) {
            return true;
        }

        assert(isfinite(radius));
        assert(isfinite(extent));

        const Point3D absDiff = (center - ballCenter).abs();

        if (absDiff.maxElement() >= radius + extent) {
            // If distance in any coordinate between centers is >= than maxDist, there is no intersection
            return false;
        }

        const Point3D diffToClosestPoint = (absDiff - extent).max(Point3D(0, 0, 0));
        return diffToClosestPoint.squaredNormL2() < radius * radius;
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
    return {hostPoints_, cudex::makeSpan(octants_), cudex::makeSpan(pointIndexes_)};
}

template<typename Point>
auto Octree<Point>::deviceIndex() -> DeviceOctreeIndex<Point>
{
    if (devicePoints_.empty()) {
        throw std::runtime_error("Device points not given");
    }

    octantsMem_.resizeCopy(cudex::makeSpan(octants_));
    pointIndexesMem_.resizeCopy(cudex::makeSpan(pointIndexes_));

    return {devicePoints_, octantsMem_.span(), pointIndexesMem_.span()};
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

    pointIndexes_.resize(hostPoints_.size());

    Point3D minValues = Point3D(hostPoints_[0]);
    Point3D maxValues = minValues;

    for (size_t i=0; i < nPoints; ++i) {
        const Point3D p = Point3D(hostPoints_[i]);

        assert(isfinite(p.x));
        assert(isfinite(p.y));
        assert(isfinite(p.z));

        minValues = minValues.min(p);
        maxValues = maxValues.max(p);

        pointIndexes_[i] = i;
    }

    const Point3D center = (minValues + maxValues) / 2;
    const Point3D extents = maxValues - center;

    assert(extents.minElement() >= 0);

    makeOctant(center, extents.maxElement() * 1.01, 0, nPoints - 1, 0);
}

template<typename Point>
Index Octree<Point>::makeOctant(
        const Point3D& center,
        const float extent,
        const Index start,
        const Index end,
        const size_t level)
{
    if (level == impl::MAX_STACK_SIZE) {
        throw std::runtime_error("Cannot build octree: too many levels");
    }

    // VLOG(2) << "Octant: " << center << ", extent: " << extent << ", size: " << (end - start + 1)
    //     << ", level: " << level;

    const Index octantIdx = octants_.size();
    octants_.emplace_back();

    assert(start <= end);
    assert(end < hostPoints_.size());

    impl::Octant o;

    o.center = center;
    o.extent = extent;
    o.start = start;
    o.end = end;

    const size_t size = end - start + 1;

    o.isLeaf = size <= params_.bucketSize || extent <= 2 * params_.minExtent;

    for (auto& c: o.children) {
        c = INVALID_INDEX;
    }

    if (o.isLeaf) {
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

    tmpIndexes_.resize(size);
    tmpCategories_.resize(size);

    // ---- Split points according to child octant

    for (size_t i = 0; i < size; ++i)
    {
        const Index pointIdx = pointIndexes_[start + i];
        assert(pointIdx != INVALID_INDEX);

        const Point3D p = Point3D(hostPoints_[pointIdx]);
        const auto childIdx = center.mortonCode(p);

        assert(childIdx < impl::N_OCTANT_CHILDREN);

        tmpCategories_[i] = childIdx;
        auto& info = childInfo[childIdx];
        info.count ++;
    }

    uint8_t lastValidChild = impl::INVALID_CHILD;
    for (size_t childIdx = 0; childIdx < impl::N_OCTANT_CHILDREN; ++childIdx) {
        auto& info = childInfo[childIdx];

        if (info.count == 0) {
            continue;
        }

        info.start = lastValidChild == impl::INVALID_CHILD ? 0 : childInfo[lastValidChild].end + 1;
        info.end = info.start + info.count - 1;

        lastValidChild = childIdx;
    }

    assert(lastValidChild != impl::INVALID_CHILD);
    assert(childInfo[lastValidChild].end + start == end);

    std::array<size_t, impl::N_OCTANT_CHILDREN> counts = {};

    for (size_t i = 0; i < size; ++i)
    {
        const uint8_t childIdx = tmpCategories_[i];
        auto& info = childInfo[childIdx];

        assert(counts[childIdx] < info.count);
        tmpIndexes_[info.start + counts[childIdx]] = pointIndexes_[start + i];
        ++counts[childIdx];
    }

    std::copy(tmpIndexes_.begin(), tmpIndexes_.end(), pointIndexes_.begin() + start);

    // ---- Create child octants

    for (size_t childIdx = 0; childIdx < impl::N_OCTANT_CHILDREN; ++childIdx) {
        const auto& info = childInfo[childIdx];

        if (info.count == 0) {
            continue;
        }

        assert(info.start != INVALID_INDEX);
        assert(info.end != INVALID_INDEX);

        const float oExtent = extent * 0.5;

        Point3D move;
        for (int d : {0, 1, 2}) {
            move[d] = ((childIdx & (1 << d)) > 0 ? 1 : -1) * oExtent;
        }

        const Point3D oCenter = center + move;

        Index& child = o.children[childIdx];
        child = makeOctant(oCenter, oExtent, start + info.start, start + info.end, level + 1);

        assert(child != INVALID_INDEX);
        assert(octants_[child].size() == info.count);
    }

    octants_[octantIdx] = o;
    return octantIdx;
}


// -------------------------------------------------------------------------------------------------
// OctreeIndex

template<bool isDevice, typename Point>
__host__ OctreeIndex<isDevice, Point>::OctreeIndex(
        Span<const Point> points,
        Span<const impl::Octant> octants,
        Span<const Index> pointIndexes)
    : points_(points)
    , octants_(octants)
    , pointIndexes_(pointIndexes)
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
            assert(octant.size() > 0);

            for (size_t i = 0; i < octant.size(); ++i)
            {
                const Index currentIndex = pointIndexes_[octant.start + i];

                const float dist2 = (query - Point3D(points_[currentIndex])).squaredNormL2();
                if (minDistance2 < dist2 && dist2 < maxDistance2)
                {
                    maxDistance2 = dist2;
                    closest = currentIndex;
                }
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
