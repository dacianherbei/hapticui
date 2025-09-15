//! Performance benchmarks for Vec3 operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use haptic::core::Vec3;

fn bench_dot_product(c: &mut Criterion) {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);

    c.bench_function("vec3_dot", |bencher| {
        bencher.iter(|| black_box(a).dot(black_box(b)))
    });
}

fn bench_cross_product(c: &mut Criterion) {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);

    c.bench_function("vec3_cross", |bencher| {
        bencher.iter(|| black_box(a).cross(black_box(b)))
    });
}

fn bench_normalize(c: &mut Criterion) {
    let v = Vec3::new(3.0, 4.0, 5.0);

    c.bench_function("vec3_normalize", |bencher| {
        bencher.iter(|| black_box(v).normalize())
    });
}

fn bench_normalize_fast(c: &mut Criterion) {
    let v = Vec3::new(3.0, 4.0, 5.0);

    c.bench_function("vec3_normalize_fast", |bencher| {
        bencher.iter(|| black_box(v).normalize_fast())
    });
}

fn bench_arithmetic(c: &mut Criterion) {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);

    c.bench_function("vec3_add", |bencher| {
        bencher.iter(|| black_box(a) + black_box(b))
    });

    c.bench_function("vec3_mul_scalar", |bencher| {
        bencher.iter(|| black_box(a) * black_box(2.5f32))
    });
}

fn bench_length_operations(c: &mut Criterion) {
    let v = Vec3::new(3.0, 4.0, 5.0);

    c.bench_function("vec3_length_squared", |bencher| {
        bencher.iter(|| black_box(v).length_squared())
    });

    c.bench_function("vec3_length", |bencher| {
        bencher.iter(|| black_box(v).length())
    });
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_cross_product,
    bench_normalize,
    bench_normalize_fast,
    bench_arithmetic,
    bench_length_operations
);

criterion_main!(benches);