// src/haptic/core/mod.rs
pub mod vec3;
pub use vec3::{Vec3, Vec4, EPSILON, SPATIAL_EPSILON};

// Your application code
use haptic::core::Vec3;

fn main() {
    let position = Vec3::new(10.0, 5.0, 2.0);
    let velocity = Vec3::new(-2.0, 0.0, 1.0);
    let new_position = position + velocity * 0.016; // 60 FPS update
}