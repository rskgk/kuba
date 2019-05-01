pub fn f32_signum(value: f32) -> f32 {
    if value == 0.0 {
        return 0.0;
    } else if value > 0.0 {
        return 1.0;
    } else {
        return -1.0;
    }
}
