//! Example usage patterns for Vec3 in HapticGUI

use haptic::core::{Vec3, EPSILON};

fn main() {
    demo_basic_operations();
    demo_spatial_transformations();
    demo_physics_simulation();
    demo_ui_positioning();
}

fn demo_basic_operations() {
    println!("=== Basic Vec3 Operations ===");

    // Create vectors
    let position = Vec3::new(10.0, 5.0, 2.0);
    let velocity = Vec3::new(-2.0, 0.0, 1.0);
    let dt = 0.016; // 60 FPS

    // Physics update
    let new_position = position + velocity * dt;
    println!("Position: {} -> {}", position, new_position);

    // Distance calculations
    let origin = Vec3::zero();
    println!("Distance from origin: {:.2}", position.distance_to(origin));

    // Normalization
    let direction = velocity.normalize();
    println!("Velocity direction: {}", direction);
    println!("Is normalized: {}", direction.is_normalized());
}

fn demo_spatial_transformations() {
    println!("\n=== Spatial Transformations ===");

    // 3D spatial operations for UI elements
    let ui_element_pos = Vec3::new(100.0, 50.0, 0.0);
    let camera_pos = Vec3::new(0.0, 0.0, 500.0);

    // Calculate view direction
    let view_dir = (ui_element_pos - camera_pos).normalize();
    println!("View direction: {}", view_dir);

    // Lighting calculations
    let light_dir = Vec3::new(0.0, 1.0, -1.0).normalize();
    let surface_normal = Vec3::new(0.0, 0.0, 1.0);

    let light_intensity = light_dir.dot(surface_normal).max(0.0);
    println!("Light intensity: {:.2}", light_intensity);

    // Reflection for glossy surfaces
    let reflected = (-light_dir).reflect(surface_normal);
    println!("Reflected light: {}", reflected);
}

fn demo_physics_simulation() {
    println!("\n=== Physics Simulation ===");

    // Simulate a bouncing ball
    let mut ball_pos = Vec3::new(0.0, 10.0, 0.0);
    let mut ball_vel = Vec3::new(5.0, 0.0, 3.0);
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let bounce_damping = 0.8;
    let dt = 0.016;

    for frame in 0..10 {
        // Apply gravity
        ball_vel += gravity * dt;

        // Update position
        ball_pos += ball_vel * dt;

        // Ground collision
        if ball_pos.y < 0.0 {
            ball_pos.y = 0.0;
            ball_vel.y = -ball_vel.y * bounce_damping;
        }

        println!("Frame {}: pos={}, vel={}", frame, ball_pos, ball_vel);
    }
}

fn demo_ui_positioning() {
    println!("\n=== UI Positioning ===");

    // Spatial UI layout in 3D
    let screen_center = Vec3::new(400.0, 300.0, 0.0);
    let panel_offset = Vec3::new(100.0, 50.0, 10.0);

    // Position UI elements relative to each other
    let main_panel = screen_center;
    let side_panel = main_panel + panel_offset;
    let button_grid_start = main_panel + Vec3::new(-50.0, -25.0, 5.0);

    println!("Main panel: {}", main_panel);
    println!("Side panel: {}", side_panel);

    // Create a grid of buttons
    for row in 0..3 {
        for col in 0..4 {
            let button_pos = button_grid_start + Vec3::new(
                col as f32 * 25.0,
                row as f32 * 20.0,
                0.0
            );
            println!("Button[{},{}]: {}", row, col, button_pos);
        }
    }

    // Smooth animation between positions
    let start_pos = main_panel;
    let end_pos = side_panel;

    println!("\nAnimation sequence:");
    for i in 0..11 {
        let t = i as f32 / 10.0;
        let current_pos = start_pos.lerp(end_pos, t);
        println!("t={:.1}: {}", t, current_pos);
    }
}