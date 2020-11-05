#include "gtest/gtest.h"

#include "octree_cuda/octree.cu.h"

using namespace octree_cuda;

TEST(point3d, operations)
{
    const Point3D p1 = {1, 2, 5};
    const Point3D p2 = {-1, 3, 4};

    EXPECT_EQ(p2.minElement(), -1);
    EXPECT_EQ(p2.maxElement(), 4);

    EXPECT_EQ(p1 + 2, Point3D(3, 4, 7));

    EXPECT_EQ(p1 + p2, Point3D(0, 5, 9));
    EXPECT_EQ(p1 - p2, Point3D(2, -1, 1));

    EXPECT_EQ(p2.abs(), Point3D(1, 3, 4));
    EXPECT_NE(p2.abs(), p2);
}

TEST(point3d, mortonCode)
{
    const Point3D p = {10, 10, 10};

    EXPECT_EQ(p.mortonCode(Point3D(11, 11, 5)), 3);
    EXPECT_EQ(p.mortonCode(Point3D(7, 7, 7)), 0);
    EXPECT_EQ(p.mortonCode(Point3D(17, 17, 17)), 7);
}

TEST(octant, containsBall)
{
    const impl::Octant o = { Point3D(0, 0, 0), 5 };

    EXPECT_TRUE(o.containsBall(Point3D(1, 1, 1), 3.5));
    EXPECT_TRUE(o.containsBall(Point3D(1, 1, 1), 4));

    EXPECT_FALSE(o.containsBall(Point3D(1, 2, 1), 4));
    EXPECT_FALSE(o.containsBall(Point3D(1, -2, 1), 4));
}

TEST(octant, overlapsBall)
{
    const impl::Octant o = { Point3D(0, 0, 0), 5 };

    EXPECT_TRUE(o.overlapsBall(Point3D(7, 7, 7), 3.5));
    EXPECT_TRUE(o.overlapsBall(Point3D(1, 1, 1), 4));

    EXPECT_TRUE(o.overlapsBall(Point3D(5, 6, 5), 4));
    EXPECT_TRUE(o.overlapsBall(Point3D(1, -2, 1), 4));

    EXPECT_FALSE(o.overlapsBall(Point3D(5, 6, 5), 1));
    EXPECT_TRUE(o.overlapsBall(Point3D(5, 6, 5), 1.001));
}
