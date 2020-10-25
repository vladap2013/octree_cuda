#pragma once

namespace octree_cuda {

template<size_t index, typename Point>
__host__ __device__ get(const Point& point)
{
    static_assert(index < 3);

    if constexpr(index == 0) return point.x;
    if constexpr(index == 1) return point.y;

    return point.z;
}


}

