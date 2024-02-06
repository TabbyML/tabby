//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.
mod dataset;
mod index;
mod repository;
mod utils;

use std::{io::{self, Write}, process::Stdio, sync::Arc, time::Duration};

use anyhow::Result;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tokio::{io::AsyncBufReadExt, time::Instant};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info, warn};

pub async fn scheduler<T: RepositoryAccess + 'static>(now: bool, access: T) -> Result<()> {
    if now {
        let repositories = access.list_repositories().await?;
        job_sync(&repositories);
        job_index(&repositories);
    } else {
        let access = Arc::new(access);
        let scheduler = JobScheduler::new().await?;

        // Every 10 minutes
        scheduler
            //.add(Job::new_async("* 1/10 * * * * *", move |_, _| {
            .add(Job::new_one_shot_async(Duration::from_secs(1), move |_, _| {
                let access = access.clone();
                Box::pin(async move {
                    info!("Running scheduler job...");
                    let exe = std::env::current_exe().unwrap();
                    let job_id = access
                        .create_job_run("scheduler".to_owned())
                        .await
                        .unwrap_or_default();

                    let mut child = tokio::process::Command::new(exe)
                        .arg("scheduler")
                        .arg("--now")
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
                                let _ = access
                                    .update_job_output(job_id, line + "\n", "".to_owned())
                                    .await;
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
                                let _ = access
                                    .update_job_output(job_id, "".to_owned(), line + "\n")
                                    .await;
                            }
                        });
                    }
                    if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
                        let _ = access.complete_job_run(job_id, exit_code).await;
                    } else {
                        let _ = access.complete_job_run(job_id, -1).await;
                    }
                })
            })?)
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
    println!();
}

fn job_sync(repositories: &[RepositoryConfig]) {
    println!("Syncing repositories...");
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
    println!();
}