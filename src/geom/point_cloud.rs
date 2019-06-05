use crate::geom::Bounds;
use crate::geom::Point;

pub type PointCloud2 = PointCloud<na::U2>;
pub type PointCloud3 = PointCloud<na::U3>;

#[derive(Debug)]
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
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    pub fn from_points(points: &[Point<NaD>]) -> Self {
        PointCloud::<NaD> {
            data: na::MatrixMN::<f32, NaD, na::Dynamic>::from_iterator(
                points.len(),
                points.iter().flat_map(|point| point.coords.iter()).cloned(),
            ),
        }
    }

    pub fn from_data(data: na::MatrixMN<f32, NaD, na::Dynamic>) -> Self {
        PointCloud::<NaD> { data: data }
    }

    pub fn points_iter<'a>(&'a self) -> impl Iterator<Item = Point<NaD>> + 'a {
        self.data.column_iter().map(|col| {
            // TODO(kgreenek): Figure out a way here not to make a copy.
            let coords = na::VectorN::<f32, NaD>::from_column_slice(col.as_slice());
            Point::<NaD>::from(coords)
        })
    }

    /// Returns the bounds that encapsulates all the points in the point cloud.
    pub fn bounds(&self) -> Bounds<NaD> {
        self.points_iter()
            .fold(Bounds::<NaD>::empty(), |mut bounds, point| {
                for (i, point_val) in point.iter().enumerate() {
                    if *point_val < bounds.min[i] {
                        bounds.min[i] = *point_val;
                    }
                    if *point_val > bounds.max[i] {
                        bounds.max[i] = *point_val;
                    }
                }
                bounds
            })
    }
}

impl PointCloud<na::U3> {
    // TODO(kgreenek): Make this generic on NaD and implement in the class above.
    pub fn transform(&self, transform: &na::MatrixN<f32, na::U4>) -> PointCloud<na::U3> {
        let homogeneous_mat = self
            .data
            .clone()
            .insert_fixed_rows::<na::U1>(self.data.nrows(), 1.0);
        let transformed_mat = transform * homogeneous_mat;
        let i = transformed_mat.nrows() - 1;
        PointCloud::<na::U3>::from_data(transformed_mat.remove_fixed_rows::<na::U1>(i))
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

    #[test]
    fn test_bounds() {
        let points = [kuba::point2![0.0, -1.2], kuba::point2![1.1, 1.2]];
        let point_cloud = kuba::PointCloud2::from_points(&points);
        let expected_bounds = kuba::bounds2![[0.0, -1.2], [1.1, 1.2]];
        assert!(approx::relative_eq!(point_cloud.bounds(), expected_bounds));
    }
}
