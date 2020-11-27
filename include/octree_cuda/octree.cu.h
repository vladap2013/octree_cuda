#pragma once

#include "access.cu.h"
#include "point.cu.h"

#include "cudex/memory.cu.h"
#include "cudex/uarray.cu.h"
#include "cudex/launcher.cu.h"
#include "cudex/cub.cu.h"

namespace octree_cuda {

using Index = uint32_t;
inline constexpr Index INVALID_INDEX = static_cast<Index>(-1);

namespace impl {
class Octant;
class PointInfo;
class OctantDevInfo;

template<size_t>
class BlockInfo;

using ChildCounts = cudex::UArray<Index, 8>;
}

template<bool isDevice, typename Point>
class OctreeIndex
{
public:
    using Octant = impl::Octant;
    
    template<typename T>
    using Span = cudex::Span<isDevice, T>;

    __host__ OctreeIndex(Span<const Point>, Span<const Octant>, Span<const Index>);

    __host__ __device__ Index findNeighbor(const Point& query, float minDistance = -1) const;
    __host__ __device__ Index findNeighbor(const Point3D& query, float minDistance = -1) const;

    __host__ __device__ const Point& point(Index index) const;

    __host__ __device__ size_t nPoints() const;

private:
    Span<const Point> points_;
    Span<const Octant> octants_;
    Span<const Index> pointIndexes_;
};


template<typename Point>
using HostOctreeIndex = OctreeIndex<false, Point>; 

template<typename Point>
using DeviceOctreeIndex = OctreeIndex<true, Point>; 

struct Params
{
    size_t bucketSize = 32;
    float minExtent = 0;
};

template<typename Point>
class Octree
{
public:
    Octree(const Params& params = Params());

    void initialize(cudex::HostSpan<const Point> hostPoints);
    void initialize(cudex::HostSpan<const Point> hostPoints, cudex::DeviceSpan<const Point> devicePoints);
    void initializeDevice(cudex::HostSpan<const Point> hostPoints, cudex::DeviceSpan<const Point> devicePoints);

    HostOctreeIndex<Point> hostIndex() const;
    DeviceOctreeIndex<Point> deviceIndex();

    ~Octree();

private:
    void makeOctantTree();
    Index makeOctant(const Point3D& center, float extent, Index startPoint, Index endPoint, size_t level);

    void makeOctantTreeGPU();

    template<typename T>
    friend class OctreeTest;

    void setIsLeaf(impl::Octant& octant) const;

private:
    constexpr inline static size_t MAX_OCTANTS_PER_BLOCK = 10;

private:
    Params params_;

    cudex::HostSpan<const Point> hostPoints_;
    cudex::DeviceSpan<const Point> devicePoints_;

    std::vector<Index> pointIndexes_;
    std::vector<impl::Octant> octants_;

    std::vector<Index> tmpIndexes_;
    std::vector<uint8_t> tmpCategories_;

    cudex::DeviceSpan<const Index> pointIndexesDev_;

    cudex::DeviceMemory<Index> pointIndexesMem_[2];
    cudex::DeviceMemory<impl::Octant> octantsMem_;

    cudex::DeviceMemory<impl::PointInfo> tmpPointInfosMem_;
    cudex::DeviceMemory<Index> tmpPointOctantMem_;
    cudex::DeviceMemory<impl::BlockInfo<MAX_OCTANTS_PER_BLOCK>> tmpBlockInfosMem_;
    cudex::DeviceMemory<impl::OctantDevInfo> tmpOctantDevInfoMem_;
    cudex::DeviceMemory<Index> tmpOctantChildIdsMem_;

    cudex::HostDeviceMemory<impl::ChildCounts> tmpChildrenMem_;
    cudex::HostDeviceMemory<Point3D> minMaxPointsMem_;

    cudex::Reduce reduce_;
};


} // namespace octree_cuda

#include "octree.inl.cu.h"
