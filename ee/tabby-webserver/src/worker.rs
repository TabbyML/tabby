use std::{
    net::SocketAddr,
    time::{SystemTime, UNIX_EPOCH},
};

use tokio::sync::RwLock;
use tracing::error;

#[derive(Default)]
pub struct WorkerGroup {
    workers: RwLock<Vec<String>>,
}

impl WorkerGroup {
    pub async fn select(&self) -> Option<String> {
        let workers = self.workers.read().await;
        if workers.len() > 0 {
            Some(workers[random_index(workers.len())].clone())
        } else {
            None
        }
    }

    pub async fn list(&self) -> Vec<String> {
        self.workers.read().await.clone()
    }

    pub async fn register(&self, addr: SocketAddr) -> Option<String> {
        let addr = format!("http://{}", addr);
        let mut workers = self.workers.write().await;
        if workers.len() >= 1 {
            error!("You need enterprise license to utilize more than 1 workers, please contact hi@tabbyml.com for information.");
            return None;
        }

        if !workers.contains(&addr) {
            workers.push(addr.clone());
        }
        Some(addr)
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

    #[tokio::test]
    async fn test_worker_group() {
        let wg = WorkerGroup::default();

        let addr1 = "127.0.0.1:8080".parse().unwrap();
        let addr2 = "127.0.0.2:8080".parse().unwrap();

        // Register success.
        assert!(wg.register(addr1).await.is_some());

        // Register failed, as > 1 workers requires enterprise license.
        assert!(wg.register(addr2).await.is_none());

        let workers = wg.list().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0], format!("http://{}", addr1));
    }
}
