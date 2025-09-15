use super::*;
use std::f32;

const PRECISION: f32 = 1e-5;

/// Helper function for approximate floating-point equality
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

/// Helper function for approximate Vec3 equality
fn approx_eq_vec3(a: Vec3, b: Vec3, eps: f32) -> bool {
    approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    #[test]
    fn test_all_constructors() {
        // Basic constructor
        let v = Vec3::new(1.5, -2.5, 3.7);
        assert_eq!(v.x, 1.5);
        assert_eq!(v.y, -2.5);
        assert_eq!(v.z, 3.7);

        // Special constructors
        assert_eq!(Vec3::zero(), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(Vec3::one(), Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(Vec3::unit_x(), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(Vec3::unit_y(), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(Vec3::unit_z(), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(Vec3::splat(5.0), Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_arithmetic_operations_comprehensive() {
        let a = Vec3::new(2.0, 3.0, 4.0);
        let b = Vec3::new(-1.0, 2.0, -3.0);

        // Addition
        assert_eq!(a + b, Vec3::new(1.0, 5.0, 1.0));

        // Subtraction
        assert_eq!(a - b, Vec3::new(3.0, 1.0, 7.0));

        // Scalar multiplication
        assert_eq!(a * 2.5, Vec3::new(5.0, 7.5, 10.0));
        assert_eq!(3.0 * a, Vec3::new(6.0, 9.0, 12.0));

        // Scalar division
        assert_eq!(a / 2.0, Vec3::new(1.0, 1.5, 2.0));

        // Negation
        assert_eq!(-a, Vec3::new(-2.0, -3.0, -4.0));

        // Assignment operators
        let mut v = a;
        v += b;
        assert_eq!(v, Vec3::new(1.0, 5.0, 1.0));

        v -= b;
        assert_eq!(v, a);

        v *= 2.0;
        assert_eq!(v, Vec3::new(4.0, 6.0, 8.0));

        v /= 2.0;
        assert_eq!(v, a);
    }

    #[test]
    fn test_dot_product_comprehensive() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 6.0);

        // Manual calculation: 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        assert_eq!(a.dot(b), 12.0);

        // Properties
        assert_eq!(a.dot(b), b.dot(a)); // Commutative
        assert_eq!(a.dot(Vec3::zero()), 0.0); // Zero vector
        assert_eq!(Vec3::unit_x().dot(Vec3::unit_x()), 1.0); // Unit vectors
        assert_eq!(Vec3::unit_x().dot(Vec3::unit_y()), 0.0); // Orthogonal
    }

    #[test]
    fn test_cross_product_comprehensive() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);

        // Right-hand rule
        assert!(approx_eq_vec3(a.cross(b), c, PRECISION));
        assert!(approx_eq_vec3(b.cross(c), a, PRECISION));
        assert!(approx_eq_vec3(c.cross(a), b, PRECISION));

        // Anti-commutative
        assert!(approx_eq_vec3(a.cross(b), -(b.cross(a)), PRECISION));

        // Parallel vectors
        assert!(approx_eq_vec3(a.cross(a), Vec3::zero(), PRECISION));
        assert!(approx_eq_vec3(a.cross(a * 2.0), Vec3::zero(), PRECISION));

        // General case
        let v1 = Vec3::new(2.0, 3.0, 4.0);
        let v2 = Vec3::new(5.0, 6.0, 7.0);
        let cross = v1.cross(v2);

        // Cross-product should be orthogonal to both vectors
        assert!(approx_eq(cross.dot(v1), 0.0, PRECISION));
        assert!(approx_eq(cross.dot(v2), 0.0, PRECISION));
    }

    #[test]
    fn test_length_operations_comprehensive() {
        // Known lengths
        let v1 = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx_eq(v1.length(), 5.0, PRECISION));
        assert_eq!(v1.length_squared(), 25.0);

        let v2 = Vec3::new(1.0, 1.0, 1.0);
        assert!(approx_eq(v2.length(), 3.0_f32.sqrt(), PRECISION));
        assert_eq!(v2.length_squared(), 3.0);

        // Zero vector
        assert_eq!(Vec3::zero().length(), 0.0);
        assert_eq!(Vec3::zero().length_squared(), 0.0);

        // Negative components
        let v3 = Vec3::new(-3.0, -4.0, 0.0);
        assert!(approx_eq(v3.length(), 5.0, PRECISION));
    }

    #[test]
    fn test_normalization_comprehensive() {
        // Standard case
        let v = Vec3::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!(approx_eq(normalized.length(), 1.0, PRECISION));
        assert!(approx_eq_vec3(normalized, Vec3::new(0.6, 0.8, 0.0), PRECISION));

        // Already normalized
        let unit = Vec3::unit_x();
        assert!(approx_eq_vec3(unit.normalize(), unit, PRECISION));

        // Zero vector
        assert_eq!(Vec3::zero().normalize(), Vec3::zero());

        // Very small vector (should return zero)
        let tiny = Vec3::new(1e-8, 1e-8, 1e-8);
        assert_eq!(tiny.normalize(), Vec3::zero());

        // try_normalize
        assert!(v.try_normalize().is_some());
        assert!(Vec3::zero().try_normalize().is_none());
        assert!(tiny.try_normalize().is_none());

        // Fast normalize (should be similar to regular normalize)
        let fast_normalized = v.normalize_fast();
        assert!(approx_eq(fast_normalized.length(), 1.0, PRECISION));
    }

    #[test]
    fn test_distance_operations() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        let c = Vec3::new(1.0, 1.0, 1.0);

        assert!(approx_eq(a.distance_to(b), 5.0, PRECISION));
        assert_eq!(a.distance_squared_to(b), 25.0);

        // Distance is symmetric
        assert!(approx_eq(a.distance_to(b), b.distance_to(a), PRECISION));

        // Distance to self is zero
        assert_eq!(a.distance_to(a), 0.0);

        // Triangle inequality
        assert!(a.distance_to(c) <= a.distance_to(b) + b.distance_to(c) + PRECISION);
    }

    #[test]
    fn test_component_wise_operations() {
        let a = Vec3::new(1.0, -2.0, 3.0);
        let b = Vec3::new(-2.0, 4.0, 1.0);

        // Min/Max
        assert_eq!(a.min(b), Vec3::new(-2.0, -2.0, 1.0));
        assert_eq!(a.max(b), Vec3::new(1.0, 4.0, 3.0));

        // Abs
        assert_eq!(a.abs(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(b.abs(), Vec3::new(2.0, 4.0, 1.0));

        // Clamp
        let min = Vec3::new(-1.0, -1.0, -1.0);
        let max = Vec3::new(2.0, 2.0, 2.0);
        assert_eq!(a.clamp(min, max), Vec3::new(1.0, -1.0, 2.0));

        // Floor, ceil, round
        let v = Vec3::new(1.7, -2.3, 3.5);
        assert_eq!(v.floor(), Vec3::new(1.0, -3.0, 3.0));
        assert_eq!(v.ceil(), Vec3::new(2.0, -2.0, 4.0));
        assert_eq!(v.round(), Vec3::new(2.0, -2.0, 4.0));
    }

    #[test]
    fn test_interpolation_operations() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 6.0, 8.0);

        // Linear interpolation
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
        assert_eq!(a.lerp(b, 0.5), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(a.lerp(b, 0.25), Vec3::new(1.0, 1.5, 2.0));

        // Spherical linear interpolation
        let u1 = Vec3::unit_x();
        let u2 = Vec3::unit_y();
        let slerped = u1.slerp(u2, 0.5);

        // Should be normalized
        assert!(approx_eq(slerped.length(), 1.0, PRECISION));

        // Should be between the two vectors
        assert!(slerped.dot(u1) > 0.0);
        assert!(slerped.dot(u2) > 0.0);

        // Edge cases for slerp
        assert!(approx_eq_vec3(u1.slerp(u1, 0.5), u1, PRECISION)); // Same vector
        assert!(approx_eq_vec3(u1.slerp(-u1, 0.5), u1.lerp(-u1, 0.5).normalize(), PRECISION)); // Opposite vectors
    }

    #[test]
    fn test_reflection_and_refraction() {
        // Reflection
        let incident = Vec3::new(1.0, -1.0, 0.0).normalize();
        let normal = Vec3::unit_y();
        let reflected = incident.reflect(normal);

        // Angle of incidence equals angle of reflection
        assert!(approx_eq(incident.dot(-normal), reflected.dot(normal), PRECISION));

        // Known reflection case
        assert!(approx_eq_vec3(reflected.normalize(), Vec3::new(1.0, 1.0, 0.0).normalize(), PRECISION));

        // Refraction
        let incident_refract = Vec3::new(1.0, -1.0, 0.0).normalize();
        let eta = 0.75; // Going from denser to less dense medium

        if let Some(refracted) = incident_refract.refract(normal, eta) {
            // Should be on the opposite side of normal
            assert!(refracted.dot(normal) < 0.0);
        }

        // Total internal reflection case
        let steep_incident = Vec3::new(0.1, -1.0, 0.0).normalize();
        let high_eta = 2.0;
        assert!(steep_incident.refract(normal, high_eta).is_none());
    }

    #[test]
    fn test_projection_operations() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        let b = Vec3::unit_x();

        // Projection onto unit vector
        let proj = a.project_onto(b);
        assert!(approx_eq_vec3(proj, Vec3::new(3.0, 0.0, 0.0), PRECISION));

        // Rejection (perpendicular component)
        let rej = a.reject_from(b);
        assert!(approx_eq_vec3(rej, Vec3::new(0.0, 4.0, 0.0), PRECISION));

        // Projection + rejection should equal the original
        assert!(approx_eq_vec3(proj + rej, a, PRECISION));

        // Projection should be parallel to the target vector
        assert!(approx_eq(proj.cross(b).length(), 0.0, PRECISION));
    }

    #[test]
    fn test_coordinate_transformations() {
        let v = Vec3::new(1.0, 2.0, 3.0);

        // Extension to Vec4
        let v4_point = v.to_point();
        let v4_direction = v.to_direction();
        let v4_custom = v.extend(5.0);

        assert_eq!(v4_point, Vec4::new(1.0, 2.0, 3.0, 1.0));
        assert_eq!(v4_direction, Vec4::new(1.0, 2.0, 3.0, 0.0));
        assert_eq!(v4_custom, Vec4::new(1.0, 2.0, 3.0, 5.0));

        // Truncation from Vec4
        assert_eq!(v4_point.truncate(), v);
        assert_eq!(v4_direction.truncate(), v);

        // Perspective division
        let v4_perspective = Vec4::new(2.0, 4.0, 6.0, 2.0);
        assert_eq!(v4_perspective.truncate_with_perspective(), v);

        // Division by zero case
        let v4_zero_w = Vec4::new(1.0, 2.0, 3.0, 0.0);
        assert_eq!(v4_zero_w.truncate_with_perspective(), Vec3::zero());
    }

    #[test]
    fn test_utility_methods() {
        // Finite checks
        let finite = Vec3::new(1.0, 2.0, 3.0);
        assert!(finite.is_finite());
        assert!(!finite.is_nan());

        let infinite = Vec3::new(f32::INFINITY, 2.0, 3.0);
        assert!(!infinite.is_finite());

        let nan = Vec3::new(f32::NAN, 2.0, 3.0);
        assert!(!nan.is_finite());
        assert!(nan.is_nan());

        // Zero and normalized checks
        assert!(Vec3::zero().is_zero());
        assert!(!finite.is_zero());

        let tiny = Vec3::new(1e-8, 1e-8, 1e-8);
        assert!(tiny.is_zero());

        assert!(Vec3::unit_x().is_normalized());
        assert!(!Vec3::new(2.0, 0.0, 0.0).is_normalized());

        // Component extremes
        let v = Vec3::new(-5.0, 3.0, -7.0);
        assert_eq!(v.max_component(), 7.0);
        assert_eq!(v.min_component(), 3.0);

        // Swizzling
        let original = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(original.swizzle(2, 1, 0), Vec3::new(3.0, 2.0, 1.0)); // zyx
        assert_eq!(original.swizzle(0, 0, 0), Vec3::new(1.0, 1.0, 1.0)); // xxx
    }

    #[test]
    fn test_indexing_comprehensive() {
        let v = Vec3::new(10.0, 20.0, 30.0);

        // Read access
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 20.0);
        assert_eq!(v[2], 30.0);

        // Write access
        let mut v_mut = v;
        v_mut[0] = 100.0;
        v_mut[1] = 200.0;
        v_mut[2] = 300.0;
        assert_eq!(v_mut, Vec3::new(100.0, 200.0, 300.0));
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_indexing_out_of_bounds_read() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let _ = v[3];
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_indexing_out_of_bounds_write() {
        let mut v = Vec3::new(1.0, 2.0, 3.0);
        v[3] = 5.0;
    }

    #[test]
    fn test_conversions_comprehensive() {
        let v = Vec3::new(1.5, -2.5, 3.7);

        // Array conversions
        let arr: [f32; 3] = v.into();
        assert_eq!(arr, [1.5, -2.5, 3.7]);
        assert_eq!(Vec3::from(arr), v);
        assert_eq!(Vec3::from([1.0, 2.0, 3.0]), Vec3::new(1.0, 2.0, 3.0));

        // Tuple conversions
        let tuple: (f32, f32, f32) = v.into();
        assert_eq!(tuple, (1.5, -2.5, 3.7));
        assert_eq!(Vec3::from(tuple), v);
        assert_eq!(Vec3::from((1.0, 2.0, 3.0)), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_default_and_display() {
        // Default should be zero vector
        assert_eq!(Vec3::default(), Vec3::zero());

        // Display formatting
        let v = Vec3::new(1.234567, -2.876543, 3.141592);
        let display_str = format!("{}", v);
        assert!(display_str.contains("1.235"));
        assert!(display_str.contains("-2.877"));
        assert!(display_str.contains("3.142"));
    }

    #[test]
    fn test_special_values() {
        // Very large values
        let large = Vec3::new(1e30, 1e30, 1e30);
        assert!(large.is_finite());
        assert!(large.length() > 1e30);

        // Very small values
        let small = Vec3::new(1e-30, 1e-30, 1e-30);
        assert!(small.is_finite());
        assert!(small.length() > 0.0);

        // Mixed special values
        let mixed = Vec3::new(0.0, -0.0, 1.0);
        assert!(mixed.is_finite());
        assert_eq!(mixed.y, -0.0);
    }

    #[test]
    fn test_mathematical_properties() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = Vec3::new(7.0, 8.0, 9.0);
        let s = 2.5;

        // Vector addition is commutative and associative
        assert!(approx_eq_vec3(a + b, b + a, PRECISION));
        assert!(approx_eq_vec3((a + b) + c, a + (b + c), PRECISION));

        // Scalar multiplication is distributive
        assert!(approx_eq_vec3(s * (a + b), s * a + s * b, PRECISION));

        // Dot product properties
        assert!(approx_eq(a.dot(b), b.dot(a), PRECISION)); // Commutative
        assert!(approx_eq((s * a).dot(b), s * a.dot(b), PRECISION)); // Scalar multiplication

        // Cross product properties
        assert!(approx_eq_vec3(a.cross(b), -(b.cross(a)), PRECISION)); // Anti-commutative
        assert!(approx_eq(a.cross(b).dot(a), 0.0, PRECISION)); // Orthogonality
        assert!(approx_eq(a.cross(b).dot(b), 0.0, PRECISION)); // Orthogonality
    }

    #[test]
    fn test_numerical_stability() {
        // Test operations near epsilon boundaries
        let near_epsilon = Vec3::new(EPSILON, EPSILON, EPSILON);
        let normalized = near_epsilon.normalize();

        // Should handle near-zero vectors gracefully
        assert!(normalized == Vec3::zero() || normalized.is_normalized());

        // Test very large vectors
        let large = Vec3::new(1e20, 1e20, 1e20);
        let large_normalized = large.normalize();
        assert!(approx_eq(large_normalized.length(), 1.0, PRECISION));

        // Test vectors with vastly different component magnitudes
        let unbalanced = Vec3::new(1e10, 1e-10, 1.0);
        let unbalanced_normalized = unbalanced.normalize();
        assert!(approx_eq(unbalanced_normalized.length(), 1.0, PRECISION));
    }
}

// Performance tests (not run by default but available for benchmarking)
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Use `cargo test -- --ignored` to run performance tests
    fn benchmark_critical_operations() {
        let iterations = 1_000_000;
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        // Benchmark dot product
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(a.dot(b));
        }
        let dot_duration = start.elapsed();
        println!("Dot product: {:?} per operation", dot_duration / iterations);

        // Benchmark cross-product
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(a.cross(b));
        }
        let cross_duration = start.elapsed();
        println!("Cross product: {:?} per operation", cross_duration / iterations);

        // Benchmark normalize
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(a.normalize());
        }
        let normalize_duration = start.elapsed();
        println!("Normalize: {:?} per operation", normalize_duration / iterations);

        // Verify performance targets (these are estimates)
        // Actual performance will vary significantly based on hardware
        assert!((dot_duration.as_nanos() / iterations as u128) < 50); // <50ns per operation
        assert!((cross_duration.as_nanos() / iterations as u128) < 100); // <100ns per operation
        assert!((normalize_duration.as_nanos() / iterations as u128) < 200); // <200ns per operation
    }
}