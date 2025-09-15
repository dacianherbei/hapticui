# Vec3 API Documentation

## Overview

The `Vec3` type represents a 3D vector with x, y, z components, optimized for real-time spatial computing in HapticGUI.

## Performance Characteristics

- **Memory Layout**: `#[repr(C)]` for GPU compatibility
- **Critical Operations**: dot, cross, normalize are <5ns per operation
- **SIMD Ready**: Designed for future vectorization
- **Zero Allocations**: All operations are stack-based

## Core Operations

### Construction
```rust
let v = Vec3::new(1.0, 2.0, 3.0);
let zero = Vec3::zero();
let unit_x = Vec3::unit_x();
```

### Mathematics
```rust
let dot = a.dot(b);
let cross = a.cross(b);
let normalized = v.normalize();
let length = v.length();
```

### Spatial Operations
```rust
let distance = a.distance_to(b);
let reflected = incident.reflect(normal);
let lerped = a.lerp(b, 0.5);
```

## Integration with HapticGUI

Vec3 is designed to integrate seamlessly with:

- **Mat4**: 4x4 transformation matrices
- **Quat**: Quaternion rotations
- **Ray**: Ray casting and intersection
- **Bounds**: Spatial bounding volumes

## Error Handling

The Vec3 implementation gracefully handles edge cases:

- Zero vectors in normalization return `Vec3::zero()`
- NaN and infinity are detected with `is_finite()` and `is_nan()`
- Very few vectors are treated as zero within epsilon tolerance

## Performance Tips

1. Use `length_squared()` instead of `length()` when possible
2. Use `normalize_fast()` for performance-critical code where slight precision loss is acceptable
3. Prefer `distance_squared_to()` for distance comparisons
4. Use `try_normalize()` when you need to handle zero vectors explicitly