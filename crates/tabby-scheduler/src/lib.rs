//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.
mod dataset;
mod index;
mod repository;
mod utils;

use std::{process::Stdio, sync::Arc};

use anyhow::Result;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tokio::io::AsyncBufReadExt;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info, warn};

pub async fn scheduler<T: RepositoryAccess + 'static>(
    now: bool,
    access: T,
    args: &[String],
) -> Result<()> {
    if now {
        let repositories = access.list_repositories().await?;
        job_sync(&repositories);
        job_index(&repositories);
    } else {
        let args = args.to_owned();
        let access = Arc::new(access);
        let scheduler = JobScheduler::new().await?;
        let scheduler_mutex = Arc::new(tokio::sync::Mutex::new(()));

        // Every 10 minutes
        scheduler
            .add(Job::new_async(
                "0 1/10 * * * *",
                move |uuid, mut scheduler| {
                    let access = access.clone();
                    let args = args.clone();
                    let scheduler_mutex = scheduler_mutex.clone();
                    Box::pin(async move {
                        let Ok(_guard) = scheduler_mutex.try_lock() else {
                            warn!("Scheduler job overlapped, skipping...");
                            return;
                        };

                        info!("Running scheduler job...");
                        let exe = std::env::current_exe().unwrap();
                        let job_id = access
                            .create_job_run("scheduler".to_owned())
                            .await
                            .unwrap_or_default();

                        let mut child = tokio::process::Command::new(exe)
                            .arg("scheduler")
                            .arg("--now")
                            .args(args)
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .spawn()
                            .unwrap();

                        {
                            // Pipe stdout
                            let access = access.clone();
                            let stdout = child.stdout.take().unwrap();
                            tokio::spawn(async move {
                                let stdout = tokio::io::BufReader::new(stdout);
                                let mut stdout = stdout.lines();
                                while let Ok(Some(line)) = stdout.next_line().await {
                                    println!("{line}");
                                    let _ = access.update_job_stdout(job_id, line + "\n").await;
                                }
                            });
                        }

                        {
                            // Pipe stderr
                            let access = access.clone();
                            let stderr = child.stderr.take().unwrap();
                            tokio::spawn(async move {
                                let stderr = tokio::io::BufReader::new(stderr);
                                let mut stdout = stderr.lines();
                                while let Ok(Some(line)) = stdout.next_line().await {
                                    eprintln!("{line}");
                                    let _ = access.update_job_stderr(job_id, line + "\n").await;
                                }
                            });
                        }
                        if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
                            let _ = access.complete_job_run(job_id, exit_code).await;
                        } else {
                            let _ = access.complete_job_run(job_id, -1).await;
                        }

                        if let Ok(Some(next_tick)) = scheduler.next_tick_for_job(uuid).await {
                            info!(
                                "Next time for scheduler job is {:?}",
                                next_tick.with_timezone(&chrono::Local)
                            );
                        }
                    })
                },
            )?)
            .await?;

        info!("Scheduler activated...");
        scheduler.start().await?;

        // Sleep 10 years (indefinitely)
        tokio::time::sleep(tokio::time::Duration::from_secs(3600 * 24 * 365 * 10)).await;
    }

    Ok(())
}

fn job_index(repositories: &[RepositoryConfig]) {
    println!("Indexing repositories...");
    let ret = index::index_repositories(repositories);
    if let Err(err) = ret {
        error!("Failed to index repositories, err: '{}'", err);
    }
}

fn job_sync(repositories: &[RepositoryConfig]) {
    println!("Syncing {} repositories...", repositories.len());
    let ret = repository::sync_repositories(repositories);
    if let Err(err) = ret {
        error!("Failed to sync repositories, err: '{}'", err);
        return;
    }

    println!("Building dataset...");
    let ret = dataset::create_dataset(repositories);
    if let Err(err) = ret {
        error!("Failed to build dataset, err: '{}'", err);
    }
}
