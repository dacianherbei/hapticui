#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPSILON: f32 = 1e-5;

    fn assert_vec3_eq(a: Vec3, b: Vec3) {
        assert!((a - b).length() < TEST_EPSILON, "Expected {:?}, got {:?}", b, a);
    }

    #[test]
    fn test_constructors() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);

        assert_eq!(Vec3::zero(), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(Vec3::one(), Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(Vec3::unit_x(), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(Vec3::unit_y(), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(Vec3::unit_z(), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_basic_arithmetic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(a - b, Vec3::new(-3.0, -3.0, -3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a / 2.0, Vec3::new(0.5, 1.0, 1.5));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_dot_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_cross_product() {
        let a = Vec3::unit_x();
        let b = Vec3::unit_y();
        assert_vec3_eq(a.cross(b), Vec3::unit_z());
        assert_vec3_eq(b.cross(a), -Vec3::unit_z());
    }

    #[test]
    fn test_length_and_normalization() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < TEST_EPSILON);
        assert_eq!(v.length_squared(), 25.0);

        let normalized = v.normalize();
        assert!((normalized.length() - 1.0).abs() < TEST_EPSILON);
        assert_vec3_eq(normalized, Vec3::new(0.6, 0.8, 0.0));
    }

    #[test]
    fn test_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(b) - 5.0).abs() < TEST_EPSILON);
        assert_eq!(a.distance_squared_to(b), 25.0);
    }

    #[test]
    fn test_component_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(3.0, 1.0, 2.0);

        assert_eq!(a.min(b), Vec3::new(1.0, 1.0, 2.0));
        assert_eq!(a.max(b), Vec3::new(3.0, 2.0, 3.0));
        assert_eq!(Vec3::new(-1.0, 2.0, -3.0).abs(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_interpolation() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 4.0, 6.0);

        assert_vec3_eq(a.lerp(b, 0.5), Vec3::new(1.0, 2.0, 3.0));
        assert_vec3_eq(a.lerp(b, 0.0), a);
        assert_vec3_eq(a.lerp(b, 1.0), b);
    }

    #[test]
    fn test_reflection() {
        let incident = Vec3::new(1.0, -1.0, 0.0);
        let normal = Vec3::unit_y();
        let reflected = incident.reflect(normal);
        assert_vec3_eq(reflected, Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn test_indexing() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);

        let mut v = v;
        v[1] = 5.0;
        assert_eq!(v[1], 5.0);
    }

    #[test]
    fn test_edge_cases() {
        // Zero vector normalization
        let zero = Vec3::zero();
        assert_eq!(zero.normalize(), Vec3::zero());
        assert!(zero.try_normalize().is_none());

        // Very small vector
        let tiny = Vec3::new(1e-8, 1e-8, 1e-8);
        assert_eq!(tiny.normalize(), Vec3::zero());

        // NaN and infinity checks
        let finite = Vec3::new(1.0, 2.0, 3.0);
        assert!(finite.is_finite());
        assert!(!finite.is_nan());

        let nan_vec = Vec3::new(f32::NAN, 2.0, 3.0);
        assert!(!nan_vec.is_finite());
        assert!(nan_vec.is_nan());
    }

    #[test]
    fn test_vec4_conversion() {
        let v3 = Vec3::new(1.0, 2.0, 3.0);
        let v4_point = v3.to_point();
        let v4_dir = v3.to_direction();

        assert_eq!(v4_point, Vec4::new(1.0, 2.0, 3.0, 1.0));
        assert_eq!(v4_dir, Vec4::new(1.0, 2.0, 3.0, 0.0));

        assert_eq!(v4_point.truncate(), v3);
        assert_eq!(v4_dir.truncate(), v3);
    }

    #[test]
    fn test_conversions() {
        let v = Vec3::new(1.0, 2.0, 3.0);

        // Array conversion
        let arr: [f32; 3] = v.into();
        assert_eq!(arr, [1.0, 2.0, 3.0]);
        assert_eq!(Vec3::from(arr), v);

        // Tuple conversion
        let tuple: (f32, f32, f32) = v.into();
        assert_eq!(tuple, (1.0, 2.0, 3.0));
        assert_eq!(Vec3::from(tuple), v);
    }

    #[test]
    fn test_performance_critical_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        // These operations should be inlined and very fast
        let _dot = a.dot(b);
        let _cross = a.cross(b);
        let _normalized = a.normalize();

        // Ensure they compile and work correctly
        assert!(_dot > 0.0);
        assert!(_cross.length() > 0.0);
        assert!((_normalized.length() - 1.0).abs() < TEST_EPSILON);
    }
}