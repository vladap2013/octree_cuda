#pragma once

namespace octree_cuda::impl {

using namespace cudex;

// -------------------------------------------------------------------------------------------------
// Helper structs for kernels

struct PointInfo
{
    Index localIdx;
    uint8_t childIdx;
};

struct OctantDevInfo
{
    Index start;
    Point3D center;
    bool isLeaf;
};

template<size_t OCTANTS_PER_BLOCK>
struct BlockInfo
{
    Index octantBegin;
    Index nOctants;
    ChildCounts starts[OCTANTS_PER_BLOCK];
};

// -------------------------------------------------------------------------------------------------
// Operators

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

// -------------------------------------------------------------------------------------------------
// Kernels

// We have to use static here, because nvcc complains if inline is used with __global__
static __global__ void kernelInitIndex(DeviceSpan<Index> data, DeviceSpan<Index> pointOctants)
{
    assert(data.size() == pointOctants.size());

    const auto index = threadLinearIndex();
    if (index >= data.size())
    {
        return;
    }

    data[index] = index;
    pointOctants[index] = 0;
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


template<typename Point, size_t OCTANTS_PER_BLOCK>
__global__ void kernelCountChildPoints(
        const size_t startOctant,
        DeviceSpan<const OctantDevInfo> octantInfos,
        DeviceSpan<const Index> src,
        DeviceSpan<const Point> points,
        DeviceSpan<const Index> pointOctants,
        DeviceSpan<PointInfo> pointInfos,
        DeviceSpan<BlockInfo<OCTANTS_PER_BLOCK>> blockInfos,
        DeviceSpan<ChildCounts> childCounts)
{
    __shared__ ChildCounts localCounts[OCTANTS_PER_BLOCK];
    __shared__ OctantDevInfo localOctantInfos[OCTANTS_PER_BLOCK];
    __shared__ Index octantBegin;
    __shared__ Index octantEnd;

    assert(src.size() == points.size());
    assert(src.size() == pointInfos.size());
    assert(blockIdx.y == 0);
    assert(blockIdx.z == 0);

    const size_t blockThreadId = threadIdx.x;

    const auto index = threadLinearIndex();

    assert(blockDim.x >= OCTANTS_PER_BLOCK);

    if (blockThreadId < OCTANTS_PER_BLOCK) {
        localCounts[blockThreadId].fill(0);
    }

    if (blockThreadId == 0) {
        octantBegin = INVALID_INDEX;
        octantEnd = 0;
    }

    __syncthreads();

    Index myOctant = INVALID_INDEX;

    const Index srcIndex = index < src.size()
        ? src[index]
        : INVALID_INDEX;
    
    if (index < src.size())
    {
        assert(srcIndex != INVALID_INDEX);
        myOctant = pointOctants[srcIndex];

        if (myOctant >= startOctant) {
            atomicMin_block(&octantBegin, myOctant);
            atomicMax_block(&octantEnd, myOctant);
        }
        else {
            myOctant = INVALID_INDEX;
        }
    }

    __syncthreads();

    const Index nOctants = octantBegin != INVALID_INDEX
        ? octantEnd - octantBegin + 1
        : 0;

    const Index batchOctantBegin = octantBegin - startOctant;

    // if (nOctants > 0 && blockThreadId == 0) {
    //     printf("XX: %d %d \n", (int) startOctant, (int) nOctants); 
    // }

    assert(nOctants <= OCTANTS_PER_BLOCK);
    assert(blockDim.x >= OCTANTS_PER_BLOCK);

    if (blockThreadId < nOctants) {
        localOctantInfos[blockThreadId] = octantInfos[batchOctantBegin + blockThreadId];
    }

    __syncthreads();

    assert(nOctants > 0 || myOctant == INVALID_INDEX);
    if (myOctant != INVALID_INDEX)
    {
        assert(srcIndex != INVALID_INDEX);

        const auto p = Point3D(points[srcIndex]);

        const Index localOctant = myOctant - octantBegin;
        assert(localOctant < nOctants);

        const auto childIdx = localOctantInfos[localOctant].center.mortonCode(p);
        assert(childIdx < 8);

        pointInfos[index].localIdx = atomicAdd_block(
            &localCounts[localOctant][childIdx],
            1
        );
        pointInfos[index].childIdx = childIdx;
    }

    __syncthreads();

    assert(blockIdx.y == 0);
    assert(blockIdx.z == 0);
    assert(blockIdx.x < blockInfos.size());

    auto& binfo = blockInfos[blockIdx.x];

    if (blockThreadId < nOctants) {
        for (size_t i : CHILD_INDEXES) {
            const Index count = localCounts[blockThreadId][i];
            const Index octant = batchOctantBegin + blockThreadId;

            binfo.starts[blockThreadId][i] = count > 0
                ? atomicAdd(&childCounts[octant][i], count)
                : INVALID_INDEX;
        }
    }

    if (blockThreadId == 0) {
        binfo.octantBegin = octantBegin;
        binfo.nOctants = nOctants;
    }
}

template<size_t OCTANTS_PER_BLOCK>
__global__ void kernelWritePoints(
        const Index startOctant,
        DeviceSpan<const OctantDevInfo> octantInfos,
        DeviceSpan<const Index> src,
        DeviceSpan<Index> dst,
        DeviceSpan<const PointInfo> pointInfos,
        DeviceSpan<const Index> pointOctants,
        DeviceSpan<const BlockInfo<OCTANTS_PER_BLOCK>> blockInfos,
        DeviceSpan<const ChildCounts> childCounts)
{
    __shared__ ChildCounts localChildCounts[OCTANTS_PER_BLOCK];
    __shared__ OctantDevInfo localOctantInfos[OCTANTS_PER_BLOCK];
    __shared__ BlockInfo<OCTANTS_PER_BLOCK> binfo;

    assert(src.size() == dst.size());
    assert(src.size() == pointInfos.size());

    const size_t blockThreadId = threadIdx.x;

    assert(blockDim.x >= OCTANTS_PER_BLOCK);

    if (blockThreadId == 0) {
        binfo = blockInfos[blockIdx.x];
        assert(binfo.nOctants <= OCTANTS_PER_BLOCK);

        for (size_t i = 0; i < binfo.nOctants; ++i)
        {
            const Index batchOctantBegin = binfo.octantBegin - startOctant;

            localChildCounts[i] = childCounts[batchOctantBegin + i];
            localOctantInfos[i] = octantInfos[batchOctantBegin + i];
        }
    }

    __syncthreads();

    const auto index = threadLinearIndex();
    const auto index2 = index;
    if (index < src.size()) {
        const Index srcIndex = src[index];
        const Index octant = pointOctants[srcIndex];

        Index dstIndex = index;

        if (octant >= binfo.octantBegin) {
            const Index localOctant = octant - binfo.octantBegin;
            assert(localOctant < OCTANTS_PER_BLOCK);

            const PointInfo pi = pointInfos[index];
            assert(pi.childIdx < 8);

            Index octantChildStart = 0;
            for (size_t i=0; i < pi.childIdx; ++i) {
                octantChildStart += localChildCounts[localOctant][i];
            }

            dstIndex = localOctantInfos[localOctant].start +
                octantChildStart +
                binfo.starts[localOctant][pi.childIdx] +
                pi.localIdx;
        }

        if (dst[dstIndex] != 0) {
            printf("ZZ1: %d %d %d %d\n", (int) index2, (int) srcIndex, (int) dstIndex, (int) dst[dstIndex]);
        }

        // assert(dst[dstIndex] == 0);
        dst[dstIndex] = srcIndex;
    }
}

template<size_t OCTANTS_PER_BLOCK>
__global__ void kernelUpdatePointOctant(
        const Index startOctant,
        DeviceSpan<Index> pointOctants,
        DeviceSpan<const Index> src,
        DeviceSpan<const Index> octantChildIds,
        DeviceSpan<const OctantDevInfo> octantInfos,
        DeviceSpan<const PointInfo> pointInfos,
        DeviceSpan<const BlockInfo<OCTANTS_PER_BLOCK>> blockInfos,
        DeviceSpan<const ChildCounts> childCount)
{
    __shared__ BlockInfo<OCTANTS_PER_BLOCK> binfo;
    __shared__ ChildCounts localChildCounts[OCTANTS_PER_BLOCK];
    __shared__ Index localOctantChildIds[OCTANTS_PER_BLOCK];
    __shared__ OctantDevInfo localOctantInfos[OCTANTS_PER_BLOCK];

    const size_t blockThreadId = threadIdx.x;

    if (blockThreadId == 0) {
        binfo = blockInfos[blockIdx.x];
        assert(binfo.nOctants <= OCTANTS_PER_BLOCK);

        for (size_t i = 0; i < binfo.nOctants; ++i)
        {
            const Index batchOctantBegin = binfo.octantBegin - startOctant;
            const Index oi = batchOctantBegin + i;

            localChildCounts[i] = childCount[oi];
            localOctantChildIds[i] = octantChildIds[oi];
            localOctantInfos[i] = octantInfos[oi];
        }
    }

    __syncthreads();

    const auto index = threadLinearIndex();
    if (index < pointInfos.size()) {
        const Index srcIndex = src[index];
        const Index parentOctant = pointOctants[srcIndex];
        const PointInfo& pi = pointInfos[index];

        assert(parentOctant != INVALID_INDEX);

        const Index oi = parentOctant - startOctant;

        if (parentOctant >= startOctant && !localOctantInfos[oi].isLeaf)
        {
            assert(binfo.octantBegin <= parentOctant && parentOctant < binfo.octantBegin + binfo.nOctants);

            const Index loi = parentOctant - binfo.octantBegin;

            Index newOctant = localOctantChildIds[loi];
            assert(newOctant != INVALID_INDEX);

            for (size_t i=0; i < pi.childIdx; ++i) {
                newOctant += localChildCounts[loi][i] > 0 ? 1 : 0;
            }

            pointOctants[srcIndex] = newOctant;
        }
    }
}



} // namespace
