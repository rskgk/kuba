pub type Point<D> = na::Point<f32, D>;
pub type Point2 = Point<na::U2>;
pub type Point3 = Point<na::U3>;

#[macro_export]
macro_rules! point2 {
    ($($val: expr),+) => {{
        $crate::Point2::new($($val),*)
    }}
}

#[macro_export]
macro_rules! point3 {
    ($($val: expr),+) => {{
        $crate::Point3::new($($val),*)
    }}
}
