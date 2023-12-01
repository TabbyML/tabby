use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tracing::error;

use crate::schema::worker::Worker;

#[derive(Default)]
pub struct WorkerGroup {
    workers: RwLock<Vec<Worker>>,
}

impl WorkerGroup {
    pub async fn select(&self) -> Option<String> {
        let workers = self.workers.read().await;
        if workers.len() > 0 {
            Some(workers[random_index(workers.len())].addr.clone())
        } else {
            None
        }
    }

    pub async fn list(&self) -> Vec<Worker> {
        self.workers.read().await.clone()
    }

    pub async fn register(&self, worker: Worker) -> Option<Worker> {
        let mut workers = self.workers.write().await;
        if workers.len() >= 1 {
            error!("You need enterprise license to utilize more than 1 workers, please contact hi@tabbyml.com for information.");
            return None;
        }

        if workers.iter().all(|x| x.addr != worker.addr) {
            workers.push(worker.clone());
        }

        Some(worker)
    }

    pub async fn unregister(&self, worker_addr: &str) -> bool {
        let mut workers = self.workers.write().await;
        if let Some(index) = workers.iter().position(|x| x.addr == worker_addr) {
            workers.remove(index);
            true
        } else {
            false
        }
    }
}

fn random_index(size: usize) -> usize {
    let unix_timestamp = (SystemTime::now().duration_since(UNIX_EPOCH))
        .unwrap()
        .as_nanos();
    let index = unix_timestamp % (size as u128);
    index as usize
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::schema::worker::WorkerKind;

    #[tokio::test]
    async fn test_worker_group() {
        let wg = WorkerGroup::default();

        let worker1 = make_worker("http://127.0.0.1:8080");
        let worker2 = make_worker("http://127.0.0.2:8080");

        // Register success.
        assert!(wg.register(worker1.clone()).await.is_some());

        // Register failed, as > 1 workers requires enterprise license.
        assert!(wg.register(worker2).await.is_none());

        let workers = wg.list().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].addr, worker1.addr);
    }

    fn make_worker(addr: &str) -> Worker {
        Worker {
            name: "Fake worker".to_owned(),
            kind: WorkerKind::Chat,
            addr: addr.to_owned(),
            device: "cuda".to_owned(),
            arch: "x86_64".to_owned(),
            cpu_info: "Fake CPU".to_owned(),
            cpu_count: 32,
            cuda_devices: vec![],
        }
    }
}
