#pragma once

#include "access.cu.h"
#include "point.cu.h"

#include "cudex/memory.cu.h"

namespace octree_cuda {

using Index = uint32_t;
inline constexpr Index INVALID_INDEX = static_cast<Index>(-1);

namespace impl {
class Octant;
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
    Span<const Index> successors_;
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
    Index makeOctant(const Point3D& center, float extent, Index startPoint, Index endPoint, size_t size);

    template<typename T>
    friend class OctreeTest;

private:
    Params params_;

    cudex::HostSpan<const Point> hostPoints_;
    cudex::DeviceSpan<const Point> devicePoints_;

    std::vector<Index> successors_;
    std::vector<impl::Octant> octants_;

    cudex::DeviceMemory<Index> successorsMem_;
    cudex::DeviceMemory<impl::Octant> octantsMem_;
};


} // namespace octree_cuda

#include "octree.inl.cu.h"
