#include "cudex/uarray.cu.h"
#include "cudex/stack.cu.h"
#include "cudex/device_utils.cu.h"

// -------------------------------------------------------------------------------------------------
// Impl

namespace octree_cuda::impl {

using namespace cudex;

using OctantIndex = Index;

constexpr inline size_t MAX_STACK_SIZE = 32;

constexpr inline uint8_t INVALID_CHILD = 255;
constexpr inline uint8_t N_OCTANT_CHILDREN = 8;

const inline static uint8_t __device__ CHILD_INDEXES[] = {0, 1, 2, 3, 4, 5, 6, 7};

using ChildCounts = cudex::UArray<Index, 8>;

struct PointInfo
{
    Index localIdx;
    uint8_t childIdx;
};

struct Octant
{
    Point3D center;
    float extent;

    Index start;
    size_t count;

    bool isLeaf;

    cudex::UArray<Index, N_OCTANT_CHILDREN> children;

    __host__ __device__ void setCenterExtent(const Point3D& minValues, const Point3D& maxValues)
    {
        center = (minValues + maxValues) / 2;

        const Point3D extents = maxValues - center;
        assert(extents.minElement() >= 0);

        extent = extents.maxElement() * 1.01;
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

// -------------------------------------------------------------------------------------------------
// CUDA

template<typename Point>
struct TransformToPoint
{
    __host__ __device__ Point3D operator()(const Point& point) const
    {
        return Point3D(point);
    }
};

struct MinElements
{
    __host__ __device__ Point3D operator()(const Point3D& p1, const Point3D& p2) const
    {
        return p1.min(p2);
    }
};

struct MaxElements
{
    __host__ __device__ Point3D operator()(const Point3D& p1, const Point3D& p2) const
    {
        return p1.max(p2);
    }
};


// We have to use static here, because nvcc complains if inline is used with __global__
static __global__ void kernelInitIndex(DeviceSpan<Index> data)
{
    const auto index = threadLinearIndex();
    if (index >= data.size())
    {
        return;
    }

    data[index] = index;
}


static __global__ void kernelInitChildCounts(DeviceSpan<ChildCounts> data)
{
    const auto index = threadLinearIndex();
    if (index >= data.size())
    {
        return;
    }

    data[index].fill(0);
}

static __global__ void copyPointIndexes(
    Index start,
    size_t count,
    DeviceSpan<const Index> src,
    DeviceSpan<Index> dst)
{
    assert(src.size() == dst.size());

    const auto index = threadLinearIndex();

    if (index >= count)
    {
        return;
    }

    const Index pi = start + index;
    dst[pi] = src[pi];
}


template<typename Point>
__global__ void kernelCountChildPoints(
        const Point3D octantCenter,
        const Index octantStart,
        const size_t octantSize,
        DeviceSpan<const Index> src,
        DeviceSpan<PointInfo> pointInfos,
        DeviceSpan<ChildCounts> blockStarts,
        ChildCounts* ptrChildCounts,
        DeviceSpan<const Point> points)
{
    __shared__ ChildCounts counts;

    const auto index = threadLinearIndex();
    if (index >= octantSize)
    {
        return;
    }

    assert(pointInfos.size() == points.size());

    const size_t thread = threadIdx.x;

    assert(blockIdx.y == 0);
    assert(blockIdx.z == 0);

    ChildCounts& childCounts = *ptrChildCounts;

    if (thread == 0) {
        counts.fill(0);
    }

    __syncthreads();

    const Index pi = octantStart + index;

    const auto p = Point3D(points[src[pi]]);
    const auto childIdx = octantCenter.mortonCode(p);

    assert(childIdx < 8);
    assert(pi < pointInfos.size());

    pointInfos[pi].localIdx = atomicAdd_block(&counts[childIdx], 1);
    pointInfos[pi].childIdx = childIdx;

    __syncthreads();

    assert(blockIdx.x < blockStarts.size());
    auto& bs = blockStarts[blockIdx.x];

    if (thread == 0) {
        for (size_t i : CHILD_INDEXES) {
            bs[i] = counts[i] > 0
                ? atomicAdd(&childCounts[i], counts[i])
                : INVALID_INDEX;
        }
    }
}

static __global__ void kernelWritePoints(
        const Index octantStart,
        const size_t octantSize,
        DeviceSpan<const Index> src,
        DeviceSpan<Index> dst,
        DeviceSpan<const PointInfo> pointInfos,
        DeviceSpan<const ChildCounts> blockStarts,
        const ChildCounts* ptrChildCounts)
{
    __shared__ ChildCounts childStarts;

    assert(src.size() == dst.size());

    const auto index = threadLinearIndex();
    if (index >= octantSize)
    {
        return;
    }

    const ChildCounts& childCounts = *ptrChildCounts;

    const size_t thread = threadIdx.x;

    if (thread == 0) {
        size_t start = 0;
        for (auto i : CHILD_INDEXES) {
            childStarts[i] = start;
            start += childCounts[i];
        }
    }

    __syncthreads();

    const Index pi = octantStart + index;
    const PointInfo& info = pointInfos[pi];

    assert(info.childIdx < 8);
    assert(pi < src.size());

    const Index position = octantStart
        + childStarts[info.childIdx]
        + blockStarts[blockIdx.x][info.childIdx]
        + info.localIdx;

    assert(position < dst.size());
    dst[position] = src[pi];
}

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

    if (pointIndexesDev_.empty()) {
        octantsMem_.resizeCopy(cudex::makeSpan(octants_));
        pointIndexesDev_ = pointIndexesMem_[0].resizeCopy(cudex::makeSpan(pointIndexes_));
    }

    return {devicePoints_, octantsMem_.span(), pointIndexesDev_};
}

template<typename Point>
void Octree<Point>::initialize(cudex::HostSpan<const Point> hostPoints)
{
    hostPoints_ = hostPoints;
    devicePoints_ = cudex::DeviceSpan<const Point>();

    if (hostPoints.empty()) {
        throw std::runtime_error("No points");
    }

    pointIndexesDev_ = cudex::DeviceSpan<const Index>();

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

    pointIndexesDev_ = cudex::DeviceSpan<const Index>();

    makeOctantTree();
}

template<typename Point>
void Octree<Point>::initializeDevice(cudex::HostSpan<const Point> hostPoints, cudex::DeviceSpan<const Point> devicePoints)
{
    if (hostPoints.empty()) {
        throw std::runtime_error("No points");
    }
    if (hostPoints.size() != devicePoints.size()) {
        throw std::runtime_error("Different sizes");
    }

    hostPoints_ = hostPoints;
    devicePoints_ = devicePoints;

    makeOctantTreeGPU();
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

    makeOctant(center, extents.maxElement() * 1.01, 0, nPoints, 0);
}

template<typename Point>
Index Octree<Point>::makeOctant(
        const Point3D& center,
        const float extent,
        const Index start,
        const Index count,
        const size_t level)
{
    if (level == impl::MAX_STACK_SIZE) {
        throw std::runtime_error("Cannot build octree: too many levels");
    }

    // VLOG(2) << "Octant: " << center << ", extent: " << extent << ", size: " << (end - start + 1)
    //     << ", level: " << level;

    const Index octantIdx = octants_.size();
    octants_.emplace_back();

    assert(start + count <= hostPoints_.size());

    impl::Octant o;

    o.center = center;
    o.extent = extent;
    o.start = start;
    o.count = count;

    o.isLeaf = count <= params_.bucketSize || extent <= params_.minExtent;

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
        size_t count = 0;
    };

    std::array<ChildInfo, impl::N_OCTANT_CHILDREN> childInfo;

    tmpIndexes_.resize(count);
    tmpCategories_.resize(count);

    // ---- Split points according to child octant

    for (size_t i = 0; i < count; ++i)
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
    for (size_t childIdx : impl::CHILD_INDEXES) {
        auto& info = childInfo[childIdx];

        if (info.count == 0) {
            continue;
        }

        info.start = lastValidChild == impl::INVALID_CHILD
            ? 0
            : childInfo[lastValidChild].start + childInfo[lastValidChild].count;

        lastValidChild = childIdx;
    }

    assert(lastValidChild != impl::INVALID_CHILD);

    std::array<size_t, impl::N_OCTANT_CHILDREN> counts = {};

    for (size_t i = 0; i < count; ++i)
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

        const float oExtent = extent * 0.5;
        const Point3D oCenter = center + oExtent * mortonCodeToOctantVector(childIdx);

        Index& child = o.children[childIdx];
        child = makeOctant(oCenter, oExtent, start + info.start, info.count, level + 1);

        assert(child != INVALID_INDEX);
        assert(octants_[child].count == info.count);
    }

    octants_[octantIdx] = o;
    return octantIdx;
}

template<typename Point>
void Octree<Point>::makeOctantTreeGPU()
{
    using namespace impl;

    const size_t nPoints = hostPoints_.size();
    assert(hostPoints_.size() == devicePoints_.size());

    DeviceSpan<Index> src = pointIndexesMem_[0].resize(nPoints);
    DeviceSpan<Index> dst = pointIndexesMem_[1].resize(nPoints);

    tmpPointInfosMem_.resize(nPoints);
    minMaxPointsMem_.resize(2);

    auto launcher = cudex::Launcher(nPoints).async();
    launcher.run(impl::kernelInitIndex, src);

    Point3D* out = minMaxPointsMem_.device().frontPtr();

    const auto initial = Point3D(hostPoints_[0]);
    reduce_.runTransformed(devicePoints_, TransformToPoint<Point>(), MinElements(), initial, out);
    reduce_.runTransformed(devicePoints_, TransformToPoint<Point>(), MaxElements(), initial, out + 1);

    syncCuda();
    minMaxPointsMem_.copyDeviceToHost();

    octants_.clear();
    octants_.emplace_back();

    octants_.back().setCenterExtent(minMaxPointsMem_[0], minMaxPointsMem_[1]);
    octants_.back().start = 0;
    octants_.back().count = nPoints;

    size_t currentStart = 0;
    size_t currentEnd = 1;
    size_t currentLevel = 0;

    while (currentStart < currentEnd) {
        if (currentLevel == impl::MAX_STACK_SIZE) {
            throw std::runtime_error("Cannot build octree: too many levels");
        }

        const size_t nOctants = currentEnd - currentStart;

        tmpChildrenMem_.resize(nOctants);

        launcher.size1D(nOctants);
        launcher.run(kernelInitChildCounts, tmpChildrenMem_.device());

        for (size_t i = 0; i < nOctants; ++i) {
            Octant& o = octants_[currentStart + i];

            o.children.fill(0);
            o.isLeaf = o.count <= params_.bucketSize || o.extent <= params_.minExtent;

            launcher.size1D(o.count);
            if (! o.isLeaf) {

                const size_t nBlocks = launcher.blockCount();
                tmpBlockMem_.resize(nBlocks);

                launcher.run(impl::kernelCountChildPoints<Point>,
                    o.center,
                    o.start,
                    o.count,
                    src,
                    tmpPointInfosMem_.span(),
                    tmpBlockMem_.span(),
                    tmpChildrenMem_.device().data() + i,
                    devicePoints_
                );

                launcher.run(impl::kernelWritePoints,
                    o.start,
                    o.count,
                    src,
                    dst,
                    tmpPointInfosMem_.cspan(),
                    tmpBlockMem_.cspan(),
                    tmpChildrenMem_.cdevice().data() + i
                );
            }
            else {
                launcher.run(impl::copyPointIndexes,
                    o.start,
                    o.count,
                    src,
                    dst
                );
            }
        }

        syncCuda();
        tmpChildrenMem_.copyDeviceToHost();

        for (size_t i = 0; i < nOctants; ++i) {
            const Index oind = currentStart + i;
            Octant& o = octants_[oind];

            if (octants_[oind].isLeaf) {
                continue;
            }

            Index start = o.start;
            for (auto childIdx : CHILD_INDEXES) {
                const auto& childCounts = tmpChildrenMem_[i];
                const size_t childCount = childCounts[childIdx];

                if (childCount == 0) {
                    octants_[oind].children[childIdx] = INVALID_INDEX;
                    continue;
                }

                octants_[oind].children[childIdx] = octants_.size();

                octants_.emplace_back();
                auto& oc = octants_.back();

                oc.extent = octants_[oind].extent * 0.5;
                oc.center = octants_[oind].center + oc.extent * mortonCodeToOctantVector(childIdx);

                oc.start = start;
                oc.count = childCount;
                start += childCount;
            }

            assert(start == octants_[oind].start + octants_[oind].count);
        }

        currentStart = currentEnd;
        currentEnd = octants_.size();
        currentLevel++;
        swap(src, dst);
    }

    pointIndexesDev_ = src;
    pointIndexes_.resize(pointIndexesDev_.size());
    cudex::copyDeviceToHost(cudex::makeSpan(pointIndexes_), pointIndexesDev_);

    octantsMem_.resizeCopy(cudex::makeSpan(octants_));
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
            assert(octant.count > 0);

            for (size_t i = 0; i < octant.count; ++i)
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
