//! High-performance 3D vector module for HapticGUI's spatial computing system.
//!
//! This module provides the foundation for all 3D spatial operations in the UI framework,
//! optimized for real-time performance with 60+ FPS requirements.

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg, Index, IndexMut};
use std::fmt;

// Constants for numerical stability
pub const EPSILON: f32 = 1e-6;
pub const SPATIAL_EPSILON: f32 = 1e-4;

/// High-performance 3D vector with x, y, z f32 components.
/// Uses #[repr(C)] for GPU compatibility and interop with graphics APIs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    // ============================================================================
    // Constructors
    // ============================================================================

    /// Creates a new Vec3 with the given components.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Creates a Vec3 with all components set to zero.
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Creates a Vec3 with all components set to one.
    #[inline]
    pub const fn one() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Creates a Vec3 representing the positive X axis.
    #[inline]
    pub const fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// Creates a Vec3 representing the positive Y axis.
    #[inline]
    pub const fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    /// Creates a Vec3 representing the positive Z axis.
    #[inline]
    pub const fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// Creates a Vec3 with all components set to the given value.
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self::new(value, value, value)
    }

    // ============================================================================
    // Basic Arithmetic Operations
    // ============================================================================

    /// Computes the dot product with another vector.
    /// Critical path operation - optimized for <5 ns performance.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product with another vector.
    /// Critical path operation - optimized for <5 ns performance.
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Computes the squared length (magnitude squared) of the vector.
    /// Faster than length() as it avoids the square root operation.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Computes the length (magnitude) of the vector.
    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Normalizes the vector to unit length.
    /// Critical path operation - optimized for <5 ns performance.
    /// Returns zero vector if the input vector is too small.
    #[inline]
    pub fn normalize(self) -> Self {
        let length_sq = self.length_squared();
        if length_sq < EPSILON * EPSILON {
            Self::zero()
        } else {
            self * fast_inv_sqrt(length_sq)
        }
    }

    /// Normalizes the vector using fast inverse square root approximation.
    /// Provides better performance for cases where slight precision loss is acceptable.
    #[inline]
    pub fn normalize_fast(self) -> Self {
        let length_sq = self.length_squared();
        if length_sq < EPSILON * EPSILON {
            Self::zero()
        } else {
            self * fast_inv_sqrt(length_sq)
        }
    }

    /// Attempts to normalize the vector, returning None if it's too close to zero.
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let length_sq = self.length_squared();
        if length_sq < EPSILON * EPSILON {
            None
        } else {
            Some(self * (1.0 / length_sq.sqrt()))
        }
    }

    /// Computes the distance to another point.
    #[inline]
    pub fn distance_to(self, other: Self) -> f32 {
        (self - other).length()
    }

    /// Computes the squared distance to another point.
    /// Faster than distance_to() as it avoids the square root operation.
    #[inline]
    pub fn distance_squared_to(self, other: Self) -> f32 {
        (self - other).length_squared()
    }

    // ============================================================================
    // Component-wise Operations
    // ============================================================================

    /// Returns the component-wise minimum of two vectors.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Returns the component-wise maximum of two vectors.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    /// Returns the component-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Clamps each component between the corresponding components of min and max.
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Returns the component-wise floor.
    #[inline]
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the component-wise ceiling.
    #[inline]
    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the component-wise round.
    #[inline]
    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round(), self.z.round())
    }

    // ============================================================================
    // Interpolation and Advanced Operations
    // ============================================================================

    /// Linearly interpolates between two vectors.
    /// t = 0.0 returns self, t = 1.0 returns other.
    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }

    /// Spherical linear interpolation between two unit vectors.
    /// Both vectors should be normalized for correct results.
    #[inline]
    pub fn slerp(self, other: Self, t: f32) -> Self {
        let dot = self.dot(other).clamp(-1.0, 1.0);

        if dot.abs() > 1.0 - EPSILON {
            // Vectors are nearly parallel, use linear interpolation
            return self.lerp(other, t).normalize();
        }

        let angle = dot.acos();
        let sin_angle = angle.sin();
        let a = ((1.0 - t) * angle).sin() / sin_angle;
        let b = (t * angle).sin() / sin_angle;

        self * a + other * b
    }

    /// Reflects the vector around a normal vector.
    /// Normal should be normalized for correct results.
    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Refracts the vector through a surface with the given normal and refractive index ratio.
    /// Returns None if total internal reflection occurs.
    #[inline]
    pub fn refract(self, normal: Self, eta: f32) -> Option<Self> {
        let cos_i = -self.dot(normal);
        let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);

        if sin_t2 > 1.0 {
            None // Total internal reflection
        } else {
            let cos_t = (1.0 - sin_t2).sqrt();
            Some(self * eta + normal * (eta * cos_i - cos_t))
        }
    }

    /// Projects this vector onto another vector.
    #[inline]
    pub fn project_onto(self, other: Self) -> Self {
        other * (self.dot(other) / other.length_squared())
    }

    /// Rejects this vector from another vector (returns the perpendicular component).
    #[inline]
    pub fn reject_from(self, other: Self) -> Self {
        self - self.project_onto(other)
    }

    // ============================================================================
    // Coordinate Space Transformations
    // ============================================================================

    /// Extends this Vec3 to a Vec4 by adding a w component.
    #[inline]
    pub fn extend(self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    /// Converts to Vec4 with w = 1.0 (point representation).
    #[inline]
    pub fn to_point(self) -> Vec4 {
        self.extend(1.0)
    }

    /// Converts to Vec4 with w = 0.0 (direction vector representation).
    #[inline]
    pub fn to_direction(self) -> Vec4 {
        self.extend(0.0)
    }

    // ============================================================================
    // Utility Methods
    // ============================================================================

    /// Checks if all components are finite (not NaN or infinite).
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Checks if any component is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Checks if the vector is approximately zero within epsilon tolerance.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.length_squared() < EPSILON * EPSILON
    }

    /// Checks if the vector is approximately normalized (unit length).
    #[inline]
    pub fn is_normalized(self) -> bool {
        (self.length_squared() - 1.0).abs() < EPSILON
    }

    /// Returns the component with the largest absolute value.
    #[inline]
    pub fn max_component(self) -> f32 {
        self.x.abs().max(self.y.abs()).max(self.z.abs())
    }

    /// Returns the component with the smallest absolute value.
    #[inline]
    pub fn min_component(self) -> f32 {
        self.x.abs().min(self.y.abs()).min(self.z.abs())
    }

    /// Returns a vector with components rearranged according to swizzle pattern.
    #[inline]
    pub fn swizzle(self, x_idx: usize, y_idx: usize, z_idx: usize) -> Self {
        Self::new(self[x_idx], self[y_idx], self[z_idx])
    }
}

// ============================================================================
// Fast Inverse Square Root Implementation
// ============================================================================

/// Fast inverse square root approximation (Quake III algorithm).
/// Provides performance improvement for normalize operations.
#[inline]
fn fast_inv_sqrt(x: f32) -> f32 {
    // For modern CPUs, the standard sqrt with division is often faster
    // than the Quake algorithm, but we provide both for compatibility
    #[cfg(feature = "fast_inv_sqrt")]
    {
        let mut i = x.to_bits();
        i = 0x5f3759df - (i >> 1);
        let y = f32::from_bits(i);
        y * (1.5 - 0.5 * x * y * y) // One Newton-Raphson iteration
    }
    #[cfg(not(feature = "fast_inv_sqrt"))]
    {
        1.0 / x.sqrt()
    }
}

// ============================================================================
// Vec4 for Extension Support
// ============================================================================

/// 4D vector for extension operations and matrix transformations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Truncates to Vec3 by dropping the w component.
    #[inline]
    pub fn truncate(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Truncates to Vec3 by performing perspective division (x/w, y/w, z/w).
    #[inline]
    pub fn truncate_with_perspective(self) -> Vec3 {
        if self.w.abs() < EPSILON {
            Vec3::zero()
        } else {
            Vec3::new(self.x / self.w, self.y / self.w, self.z / self.w)
        }
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

// Arithmetic operators
impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, vec: Vec3) -> Vec3 {
        vec * self
    }
}

impl MulAssign<f32> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        *self = *self * scalar;
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        let inv_scalar = 1.0 / scalar;
        Self::new(self.x * inv_scalar, self.y * inv_scalar, self.z * inv_scalar)
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        *self = *self / scalar;
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

// Indexing
impl Index<usize> for Vec3 {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Vec3: {}", index),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Vec3: {}", index),
        }
    }
}

// Display formatting
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

// Default (zero vector)
impl Default for Vec3 {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

// Conversion from array
impl From<[f32; 3]> for Vec3 {
    #[inline]
    fn from(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

// Conversion to array
impl From<Vec3> for [f32; 3] {
    #[inline]
    fn from(vec: Vec3) -> Self {
        [vec.x, vec.y, vec.z]
    }
}

// Conversion from tuple
impl From<(f32, f32, f32)> for Vec3 {
    #[inline]
    fn from(tuple: (f32, f32, f32)) -> Self {
        Self::new(tuple.0, tuple.1, tuple.2)
    }
}

// Conversion to tuple
impl From<Vec3> for (f32, f32, f32) {
    #[inline]
    fn from(vec: Vec3) -> Self {
        (vec.x, vec.y, vec.z)
    }
}