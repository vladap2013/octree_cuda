#pragma once

#include "access.cu.h"

namespace octree_cuda {

struct Point3D
{
    float x, y, z;

    Point3D() = default;

    Point3D(float px, float py, float pz)
        : x(px), y(py), z(pz)
    {}

    template<typename T>
    explicit Point3D(const T& point)
        : x(get<0>(point)) , y(get<1>(point)) , z(get<2>(point))
    {}

    __host__ __device__ float maxElement() const
    {
        return std::max(x, std::max(y, z));
    }

    __host__ __device__ float operator(size_t i) const
    {
        assert(i < 3);

        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
        };
    }

    template<typename F>
    __host__ __device__ Point3D elementWise(const Point3D& other, F f)
    {
        return { f(x, other.x), f(y, other.y), f(z, other.z) };
    }

    template<typename F>
    __host__ __device__ Point3D elementWise(F f)
    {
        return { f(x), f(y), f(z) };
    }

    __host__ __device__ uint8_t mortonCode(const Point3D& point)
    {
        uint8_t mortonCode = 0;

        constexpr int DIMS[] = {0, 1, 2};

        for (auto dim : DIMS) {
            if (query(dim) > octant.point(dim)) {
                mortonCode |= (1 << dim);
            }
        }

        return mortonCode;
    }

    template<typename Point>
    __host__ __device__ uint8_t mortonCode(const Point& point)
    {
        return mortonCode(Point3D(point));
    }
};

__host__ __device__ Point3D operator-(const Point3D& p1, const Point3D& p2)
{
    return {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
}

__host__ __device__ Point3D operator+(const Point3D& p1, const Point3D& p2)
{
    return {p1.x + p2.x, p1.y + p2.y, p1.z + p2.z};
}

__host__ __device__ Point3D operator*(float f, const Point3D& p)
{
    return { f * p.x, f * p.y, f * p.z };
}


}
