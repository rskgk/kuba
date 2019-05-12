use crate::geom::Point;

pub type PointCloud2 = PointCloud<na::U2>;
pub type PointCloud3 = PointCloud<na::U3>;

pub struct PointCloud<NaD>
where
    NaD: na::DimName,
{
    // The data structure that stores the points. Each point is a column in this matrix, so there
    // are as many columns as points.
    pub data: na::MatrixMN<f32, NaD, na::Dynamic>,
}

impl<NaD> PointCloud<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD>,
{
    pub fn from_points(points: &[Point<NaD>]) -> Self {
        PointCloud::<NaD> {
            data: na::MatrixMN::<f32, NaD, na::Dynamic>::from_iterator(
                points.len(),
                points.iter().flat_map(|point| point.coords.iter()).cloned(),
            ),
        }
    }

    pub fn points_iter<'a>(&'a self) -> impl Iterator<Item = Point<NaD>> + 'a {
        self.data.column_iter().map(|col| {
            // TODO(kgreenek): Figure out a way here not to make a copy.
            let coords = na::VectorN::<f32, NaD>::from_column_slice(col.as_slice());
            Point::<NaD>::from(coords)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn test_points_iter() {
        let points = [kuba::point2![0.0, 0.0], kuba::point2![1.0, 1.0]];
        let point_cloud = kuba::PointCloud2::from_points(&points);
        assert_eq!(point_cloud.points_iter().collect::<Vec<_>>(), points);
    }
}
