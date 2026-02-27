use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hx_core::{KnowledgeNode, NodeKind, QueryFilters};
use hx_engine::config::EngineConfig;
use hx_engine::engine::HelixEngine;
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn bench_sizes(default: &[usize]) -> Vec<usize> {
    if let Ok(raw) = std::env::var("HELIX_BENCH_SIZES") {
        let mut sizes = raw
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|s| *s > 0)
            .collect::<Vec<_>>();
        if !sizes.is_empty() {
            sizes.sort_unstable();
            sizes.dedup();
            return sizes;
        }
    }

    let mut sizes = default.to_vec();
    if std::env::var("HELIX_BENCH_LARGE")
        .ok()
        .as_deref()
        == Some("1")
    {
        sizes.extend([10_000, 100_000, 1_000_000]);
    }
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

fn create_engine(rt: &Runtime) -> (HelixEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let mut config = EngineConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    // Use "none" provider to avoid loading ONNX Runtime in bench environments
    config.embedding.provider = "none".into();
    let engine = rt.block_on(async { HelixEngine::init(config).await.unwrap() });
    (engine, temp_dir)
}

fn bench_engine_store_node(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (engine, _tmp) = create_engine(&rt);

    c.bench_function("engine_store_node", |b| {
        b.iter(|| {
            let node = KnowledgeNode::new(NodeKind::Fact, "Benchmark content");
            rt.block_on(async { engine.store_node(node).await.unwrap() });
        });
    });
}

fn bench_engine_get_node(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (engine, _tmp) = create_engine(&rt);

    let node = KnowledgeNode::new(NodeKind::Fact, "Test content");
    let stored = rt.block_on(async { engine.store_node(node).await.unwrap() });
    let id = stored.id;

    c.bench_function("engine_get_node", |b| {
        b.iter(|| {
            rt.block_on(async { engine.get_node(id).await.unwrap() });
        });
    });
}

fn bench_engine_list_nodes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (engine, _tmp) = create_engine(&rt);

    // Pre-populate with 100 nodes
    rt.block_on(async {
        for i in 0..100 {
            let node = KnowledgeNode::new(NodeKind::Fact, format!("Content {i}"));
            engine.store_node(node).await.unwrap();
        }
    });

    c.bench_function("engine_list_100_nodes", |b| {
        let filters = QueryFilters::default();
        b.iter(|| {
            rt.block_on(async { engine.list_nodes(&filters, 100, 0).await.unwrap() });
        });
    });
}

fn bench_engine_update_node(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (engine, _tmp) = create_engine(&rt);

    let node = KnowledgeNode::new(NodeKind::Fact, "Original content");
    let stored = rt.block_on(async { engine.store_node(node).await.unwrap() });

    c.bench_function("engine_update_node", |b| {
        let mut version = 0u32;
        b.iter(|| {
            version += 1;
            let mut updated = stored.clone();
            updated.content = format!("Updated content v{version}");
            rt.block_on(async { engine.update_node(updated).await.unwrap() });
        });
    });
}

fn bench_engine_store_1000(c: &mut Criterion) {
    let sizes = bench_sizes(&[1_000]);
    let mut group = c.benchmark_group("engine_batch_store");

    for size in sizes {
        let sample_size = if size >= 10_000 { 5 } else { 10 };
        group.sample_size(sample_size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let rt = Runtime::new().unwrap();
                b.iter(|| {
                    let (engine, _tmp) = create_engine(&rt);
                    rt.block_on(async {
                        for i in 0..size {
                            let node =
                                KnowledgeNode::new(NodeKind::Fact, format!("Batch content {i}"));
                            engine.store_node(node).await.unwrap();
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

fn bench_engine_list_in_1000(c: &mut Criterion) {
    let sizes = bench_sizes(&[1_000, 10_000]);
    let mut group = c.benchmark_group("engine_list_large");

    for size in sizes {
        let sample_size = if size >= 100_000 { 5 } else { 10 };
        group.sample_size(sample_size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let rt = Runtime::new().unwrap();
            let (engine, _tmp) = create_engine(&rt);

            rt.block_on(async {
                for i in 0..size {
                    let kind = if i % 3 == 0 {
                        NodeKind::Task
                    } else {
                        NodeKind::Fact
                    };
                    let node = KnowledgeNode::new(kind, format!("Content {i}"));
                    engine.store_node(node).await.unwrap();
                }
            });

            let filters = QueryFilters::default();
            b.iter(|| {
                rt.block_on(async { engine.list_nodes(&filters, 100, 0).await.unwrap() });
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_engine_store_node,
    bench_engine_get_node,
    bench_engine_list_nodes,
    bench_engine_update_node,
    bench_engine_store_1000,
    bench_engine_list_in_1000
);
criterion_main!(benches);
