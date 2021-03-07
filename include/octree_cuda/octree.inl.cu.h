#include "cudex/uarray.cu.h"
#include "cudex/stack.cu.h"
#include "cudex/device_utils.cu.h"

// -------------------------------------------------------------------------------------------------
// Impl

namespace octree_cuda::impl {

using OctantIndex = Index;

constexpr inline size_t MAX_STACK_SIZE = 32;

constexpr inline uint8_t INVALID_CHILD = 255;
constexpr inline uint8_t N_OCTANT_CHILDREN = 8;

const inline static uint8_t __device__ CHILD_INDEXES[] = {0, 1, 2, 3, 4, 5, 6, 7};

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

class TTimer
{
public:
    TTimer() : start_(std::chrono::high_resolution_clock::now())
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

} // namespace octree_cuda::impl

#include "octree_cuda_impl.inl.cu.h"

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

    setIsLeaf(o);

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

    for (auto childIdx : impl::CHILD_INDEXES) {
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
    TTimer timer;

    const size_t nPoints = hostPoints_.size();
    assert(hostPoints_.size() == devicePoints_.size());

    // Init device memory 

    DeviceSpan<Index> src = pointIndexesMem_[0].resize(nPoints);
    DeviceSpan<Index> dst = pointIndexesMem_[1].resize(nPoints);

    tmpPointInfosMem_.resize(nPoints);
    tmpPointOctantMem_.resize(nPoints);

    auto launcherPoints = cudex::Launcher(nPoints).async();
    launcherPoints.run(impl::kernelInitIndex,
        src,
        tmpPointOctantMem_.span()
    );

    tmpBlockInfosMem_.resize(launcherPoints.blockCount());

    // Create initial octant
    minMaxPointsMem_.resize(2);
    Point3D* out = minMaxPointsMem_.device().frontPtr();

    const auto initial = Point3D(hostPoints_[0]);
    reduce_.runTransformed(devicePoints_, TransformToPoint<Point>(), MinElements(), initial, out + 0);
    reduce_.runTransformed(devicePoints_, TransformToPoint<Point>(), MaxElements(), initial, out + 1);

    syncCuda();
    minMaxPointsMem_.copyDeviceToHost();

    octants_.clear();
    octants_.emplace_back();

    octants_.back().setCenterExtent(minMaxPointsMem_[0], minMaxPointsMem_[1]);
    octants_.back().start = 0;
    octants_.back().count = nPoints;
    setIsLeaf(octants_.back());

    // Setup main loop
    size_t currentStart = 0;
    size_t currentEnd = 1;
    size_t currentLevel = 0;

    std::vector<impl::OctantDevInfo> octantDevInfos = { {0, octants_.back().center, false} };
    std::vector<Index> octantChildIds;

    bool allLeafs = octants_.back().isLeaf;
    while (currentStart < currentEnd && !allLeafs) {
        if (currentLevel == impl::MAX_STACK_SIZE) {
            throw std::runtime_error("Cannot build octree: too many levels");
        }

        VLOG(2) << "Level: " << currentLevel << timer.printMs(", start");
        VLOG(2) << "range: " << currentStart << " - " << currentEnd;

        const size_t nOctants = currentEnd - currentStart;

        tmpChildrenMem_.resize(nOctants);

        assert(octantDevInfos.size() == nOctants);
        tmpOctantDevInfoMem_.resizeCopy(octantDevInfos);

        if (nOctants < 100) {
            size_t jj = currentStart;
            for (const auto& oi : octantDevInfos) {
                VLOG(2) << "    octantDevInfo: start: " << oi.start << ", center: " << oi.center;
                VLOG(2) << "       Octant: start: " << octants_[jj].start << ", count: " << octants_[jj].count <<  ", extent: " << octants_[jj].extent;
                ++jj;
            }
        }

        auto launcherOctants = cudex::Launcher(nOctants).async();

        launcherOctants.run(kernelInitChildCounts,
            tmpChildrenMem_.device()
        );

        launcherPoints.run(impl::kernelCountChildPoints<Point, MAX_OCTANTS_PER_BLOCK>,
            currentStart,
            tmpOctantDevInfoMem_.cspan(),
            src.cspan(),
            devicePoints_,
            tmpPointOctantMem_.cspan(),
            tmpPointInfosMem_.span(),
            tmpBlockInfosMem_.span(),
            tmpChildrenMem_.device()
        );

        cudaMemset(dst.data(), 0, dst.size_bytes());

        launcherPoints.run(impl::kernelWritePoints<MAX_OCTANTS_PER_BLOCK>,
            currentStart,
            tmpOctantDevInfoMem_.cspan(),
            src.cspan(),
            dst,
            tmpPointInfosMem_.cspan(),
            tmpPointOctantMem_.cspan(),
            tmpBlockInfosMem_.cspan(),
            tmpChildrenMem_.cdevice()
        );

        syncCuda();
        tmpChildrenMem_.copyDeviceToHost();

        octantChildIds.clear();
        octantDevInfos.clear();

        allLeafs = true;
        for (size_t i = 0; i < nOctants; ++i) {
            const Index oind = currentStart + i;
            Octant& o = octants_[oind];
            assert(o.count > 0);

            const auto& cinf = tmpChildrenMem_[i];

            Index startIndex = o.start;

            if (o.isLeaf) {
                octantChildIds.push_back(INVALID_INDEX);
                continue;
            }

            octantChildIds.push_back(octants_.size());

            for (auto childIdx : CHILD_INDEXES) {
                const Index count = cinf[childIdx];
                if (count == 0) {
                    assert(octants_[oind].children[childIdx] == INVALID_INDEX);
                    continue;
                }

                octants_[oind].children[childIdx] = octants_.size();
                octants_.emplace_back();

                auto &oc = octants_.back();

                oc.extent = octants_[oind].extent * 0.5;
                oc.center = octants_[oind].center + oc.extent * mortonCodeToOctantVector(childIdx);

                oc.count = count;
                oc.start = startIndex;
                startIndex += count;

                oc.children.fill(INVALID_INDEX);
                setIsLeaf(oc);

                if (currentLevel == 2) {
                    oc.isLeaf = true;
                }

                allLeafs = allLeafs && oc.isLeaf;

                octantDevInfos.push_back({oc.start, oc.center, oc.isLeaf});
            }

            CHECK_EQ(startIndex, o.start + o.count);
            CHECK_GT(octants_.size(), octantChildIds.back());
        }

        tmpOctantChildIdsMem_.resizeCopy(cudex::makeSpan(octantChildIds));

        launcherPoints.run(kernelUpdatePointOctant<MAX_OCTANTS_PER_BLOCK>,
            currentStart,
            tmpPointOctantMem_.span(),
            src,
            tmpOctantChildIdsMem_.cspan(),
            tmpOctantDevInfoMem_.cspan(),
            tmpPointInfosMem_.cspan(),
            tmpBlockInfosMem_.cspan(),
            tmpChildrenMem_.cdevice()
        );

        swap(src, dst);
        cudaMemset(dst.data(), dst.size_bytes(), 0);

        currentStart = currentEnd;
        currentEnd = octants_.size();
        currentLevel++;
    }

    pointIndexesDev_ = src;
    pointIndexes_.resize(pointIndexesDev_.size());
    cudex::copyDeviceToHost(cudex::makeSpan(pointIndexes_), pointIndexesDev_);

    octantsMem_.resizeCopy(cudex::makeSpan(octants_));
}

template<typename Point>
void Octree<Point>::setIsLeaf(impl::Octant& o) const
{
    o.isLeaf = o.count <= params_.bucketSize || o.extent <= params_.minExtent;
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
