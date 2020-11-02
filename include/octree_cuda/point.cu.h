#pragma once

#include "access.cu.h"

namespace octree_cuda {

struct Point3D
{
    float x, y, z;

    Point3D() = default;

    __host__ __device__ Point3D(float px, float py, float pz)
        : x(px), y(py), z(pz)
    {}

    template<typename T>
    __host__ __device__ explicit Point3D(const T& point)
        : x(get<0>(point)) , y(get<1>(point)) , z(get<2>(point))
    {}

    __host__ __device__ float maxElement() const
    {
        return fmaxf(x, fmaxf(y, z));
    }

    __host__ __device__ float minElement() const
    {
        return fminf(x, fminf(y, z));
    }

    __host__ __device__ float operator[](size_t i) const
    {
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;

        assert(false);
        return -1;
    }

    __host__ __device__ float& operator[](size_t i)
    {
        if (i == 0) return x;
        if (i == 1) return y;

        assert(i == 2);
        return z;
    }

    template<typename F>
    __host__ __device__ Point3D elementWise(F f) const
    {
        return { f(x), f(y), f(z) };
    }

    template<typename F>
    __host__ __device__ Point3D elementWise(const Point3D& other, F f) const
    {
        return { f(x, other.x), f(y, other.y), f(z, other.z) };
    }

    __host__ __device__ uint8_t mortonCode(const Point3D& point) const
    {
        uint8_t code = 0;

        constexpr int DIMS[] = {0, 1, 2};
        for (auto dim : DIMS) {
            if (point[dim] > (*this)[dim]) {
                code |= (1 << dim);
            }
        }

        return code;
    }

    __host__ __device__ Point3D abs() const
    {
        return elementWise(&fabsf);
    }

    __host__ __device__ Point3D min(const Point3D& other)
    {
        return elementWise(other, &fminf);
    }

    __host__ __device__ Point3D max(const Point3D& other)
    {
        return elementWise(other, &fmaxf);
    }

    __host__ __device__ float squaredNormL2() const
    {
        return x*x + y*y + z*z;
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

__host__ __device__ Point3D operator+(const Point3D& p, float f)
{
    return p + Point3D(f, f, f);
}

__host__ __device__ Point3D operator-(const Point3D& p, float f)
{
    return p - Point3D(f, f, f);
}
__host__ __device__ Point3D operator*(float f, const Point3D& p)
{
    return { p.x * f, p.y * f, p.z * f};
}

__host__ __device__ Point3D operator/(const Point3D& p, float f)
{
    return { p.x / f, p.y / f, p.z / f};
}

std::ostream& operator<<(std::ostream& os, const Point3D& p)
{
    os << p.x << ", " << p.y << ", " << p.z;
    return os;
}

}
