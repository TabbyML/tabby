use opentelemetry::{trace::TracerProvider as _, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    runtime,
    trace::{RandomIdGenerator, Sampler, TracerProvider},
    Resource,
};
use opentelemetry_semantic_conventions::{
    attribute::{DEPLOYMENT_ENVIRONMENT_NAME, SERVICE_NAME, SERVICE_VERSION},
    SCHEMA_URL,
};
use tracing::level_filters::LevelFilter;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{
    layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer, Registry,
};

// Create a Resource that captures information about the entity for which telemetry is recorded.
fn resource() -> Resource {
    Resource::from_schema_url(
        [
            KeyValue::new(SERVICE_NAME, env!("CARGO_PKG_NAME")),
            KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
            KeyValue::new(DEPLOYMENT_ENVIRONMENT_NAME, "develop"),
        ],
        SCHEMA_URL,
    )
}

// Construct TracerProvider for OpenTelemetryLayer
fn init_tracer_provider(otlp_endpoint: String) -> TracerProvider {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .with_timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();

    TracerProvider::builder()
        // Customize sampling strategy
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            1.0,
        ))))
        // If export trace to AWS X-Ray, you can use XrayIdGenerator
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource())
        .with_batch_exporter(exporter, runtime::Tokio)
        .build()
}

// Initialize tracing-subscriber and return OtelGuard for opentelemetry-related termination processing
pub fn init_tracing_subscriber(otlp_endpoint: Option<String>) -> OtelGuard {
    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = Vec::new();

    let tracer_provider = if let Some(endpoint) = otlp_endpoint {
        let tracer_provider = init_tracer_provider(endpoint);
        let tracer = tracer_provider.tracer("tracing-otel-subscriber");
        layers.push(Box::new(OpenTelemetryLayer::new(tracer)));
        Some(tracer_provider)
    } else {
        None
    };

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true)
        .boxed();
    layers.push(fmt_layer);

    let mut dirs = if cfg!(feature = "prod") {
        "tabby=info,otel=debug,http_api_bindings=info,llama_cpp_server=info".into()
    } else {
        "tabby=debug,otel=debug,http_api_bindings=debug,llama_cpp_server=debug".into()
    };

    if let Ok(env) = std::env::var(EnvFilter::DEFAULT_ENV) {
        dirs = format!("{dirs},{env}")
    };

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::WARN.into())
        .parse_lossy(dirs);

    tracing_subscriber::registry()
        .with(layers)
        .with(env_filter)
        .init();

    OtelGuard { tracer_provider }
}

pub struct OtelGuard {
    tracer_provider: Option<TracerProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(tracer_provider) = self.tracer_provider.take() {
            if let Err(err) = tracer_provider.shutdown() {
                eprintln!("{err:?}");
            }
        }
    }
}
