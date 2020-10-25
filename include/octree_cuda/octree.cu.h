#pragma once

#include "access.cu.h"
#include "point.cu.h"
#include "distances.cu.h"

namespace octree_cuda {

using Index = uint32_t;

namespace impl {
class Octant;
}

template<typename Point>
class GPUQuery
{
};

class Params
{
};

template<typename Point>
class Octree
{
public:
    Octree(const Params& params);

    void initialize(cudex::HostSpan<const Point> hostPoints);
    void initialize(cudex::HostSpan<const Point> hostPoints, cudex::DeviceSpan<const Point> devicePoints);
    void initializeDevice(cudex::HostSpan<const Point> hostPoints, cudex::DeviceSpan<const Point> devicePoints);

    Id findNeighbor(const Point& query, float minDistance = -1) const;

    GPUQuery<Point> gpuQuery() const;

    ~Octree();

private:
    cudex::HostSpan<const Point> hostPoints_;
    cudex::DeviceSpan<const Point> devicePoints_;

    std::vector<Index> successors_;
    std::vector<impl::Octant> octants_;

    cudex::DeviceMemory<Index> successorsMem_;
    cudex::DeviceMemory<impl::Octant> octantsMem_;
};


} // namespace octree_cuda

#include "octree.inl.cu.h"
