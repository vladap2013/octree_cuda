
namespace octree_cuda::impl {

struct Octant
{
    Point3d center;
    float extent;

    bool isLeaf;

    Index start;
    size_t size;

    Index children[8];
};


} // namespace octree_cuda::impl


namespace octree_cuda {

template<typename Point>
void Octree<Point>::initHost()
{

}



}
