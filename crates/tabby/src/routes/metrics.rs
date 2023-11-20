use std::sync::Arc;

use axum::extract::State;
use axum_prometheus::metrics_exporter_prometheus::PrometheusHandle;

pub async fn metrics(State(state): State<Arc<PrometheusHandle>>) -> String {
    state.render()
}
