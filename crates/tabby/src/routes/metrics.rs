use std::sync::Arc;
use axum::extract::State;
use axum_prometheus::metrics_exporter_prometheus::PrometheusHandle;

#[utoipa::path(
    get,
    path = "/v1/metrics",
    tag = "v1",
    responses(
        (status = 200, description = "Success", body = String, content_type = "text/plain"),
    )
)]
pub async fn metrics(State(state): State<Arc<PrometheusHandle>>) -> String {
    state.render()
}
