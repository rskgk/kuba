pub fn f32_signum(value: f32) -> f32 {
    if value == 0.0 {
        return 0.0;
    } else if value > 0.0 {
        return 1.0;
    } else {
        return -1.0;
    }
}

pub fn logodds_from_probability(probability: f32) -> f32 {
    (probability / (1.0 - probability)).ln()
}

pub fn probability_from_logodds(logodds: f32) -> f32 {
    1.0 - (1.0 / (1.0 + logodds.exp()))
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn f32_signum() {
        assert_eq!(kuba::math::f32_signum(0.0), 0.0);
        assert_eq!(kuba::math::f32_signum(0.1), 1.0);
        assert_eq!(kuba::math::f32_signum(100.1), 1.0);
        assert_eq!(kuba::math::f32_signum(-0.1), -1.0);
        assert_eq!(kuba::math::f32_signum(-100.1), -1.0);
    }

    #[test]
    fn logodds_from_probability() {
        assert_eq!(kuba::math::logodds_from_probability(0.0), -std::f32::INFINITY);
        assert_eq!(kuba::math::logodds_from_probability(1.0), std::f32::INFINITY);
        assert_eq!(kuba::math::logodds_from_probability(0.5), 0.0);
        assert_eq!(kuba::math::logodds_from_probability(0.7), 0.84729785);
        assert_eq!(kuba::math::logodds_from_probability(0.4), -0.40546516);
        assert_eq!(kuba::math::logodds_from_probability(0.1192), -2.000028);
        assert_eq!(kuba::math::logodds_from_probability(0.971), 3.5110312);
    }

    #[test]
    fn probability_from_logodds() {
        assert_eq!(kuba::math::probability_from_logodds(-std::f32::INFINITY), 0.0);
        assert_eq!(kuba::math::probability_from_logodds(std::f32::INFINITY), 1.0);
        assert_eq!(kuba::math::probability_from_logodds(0.0), 0.5);
        assert_eq!(kuba::math::probability_from_logodds(0.85), 0.7005671);
        assert_eq!(kuba::math::probability_from_logodds(-0.4), 0.40131235);
        assert_eq!(kuba::math::probability_from_logodds(-2.0), 0.11920297);
        assert_eq!(kuba::math::probability_from_logodds(3.5), 0.97068775);
    }
}
