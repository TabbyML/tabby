mod logger;

use std::{pin::Pin, str::FromStr};

use apalis::{
    cron::{CronStream, Schedule},
    prelude::{Data, Job, Storage, WorkerBuilder},
};
use chrono::{DateTime, Utc};
use futures::Stream;
pub use logger::{JobLogLayer, JobLogger};
use tabby_db::DbConn;
use tower::{
    layer::util::{Identity, Stack},
    limit::ConcurrencyLimitLayer,
};

type DefaultMiddleware =
    Stack<JobLogLayer, Stack<ConcurrencyLimitLayer, Stack<Data<DbConn>, Identity>>>;

pub trait BasicJob: Job + Sized {
    fn basic_worker<NS, Serv>(
        storage: NS,
        db: DbConn,
    ) -> WorkerBuilder<Self, NS, DefaultMiddleware, Serv>
    where
        NS: Storage<Job = Self>,
    {
        WorkerBuilder::new(Self::NAME)
            .with_storage(storage)
            .data(db.clone())
            .layer(ConcurrencyLimitLayer::new(1))
            .layer(JobLogLayer::new(db, Self::NAME))
    }
}

impl<T: Job> BasicJob for T {}

pub trait CronJob: Job {
    const CRON_NAME: &'static str = buf_and_len_to_str(&concat_buf(Self::NAME, "-cron"));
    const SCHEDULE: &'static str;

    fn cron_worker<Serv>(
        db: DbConn,
    ) -> WorkerBuilder<
        DateTime<Utc>,
        Pin<
            Box<
                (dyn Stream<
                    Item = Result<
                        std::option::Option<apalis::prelude::Request<DateTime<Utc>>>,
                        apalis::prelude::Error,
                    >,
                > + std::marker::Send
                     + 'static),
            >,
        >,
        DefaultMiddleware,
        Serv,
    > {
        let schedule = Schedule::from_str(Self::SCHEDULE).expect("invalid cron schedule");
        let stream = CronStream::new(schedule).into_stream();
        WorkerBuilder::new(Self::CRON_NAME)
            .data(db.clone())
            .stream(stream)
            .layer(ConcurrencyLimitLayer::new(1))
            .layer(JobLogLayer::new(db, Self::CRON_NAME))
    }
}

const fn buf_and_len_to_str(buf_len: &'static ([u8; 60], usize)) -> &'static str {
    let buf = &buf_len.0;
    let len = buf_len.1;
    assert!(len < buf.len(), "buf is too long");
    // I didn't find a way to slice an array in const fn
    let buf = unsafe { core::slice::from_raw_parts(buf.as_ptr(), len) };
    match core::str::from_utf8(buf) {
        Ok(s) => s,
        Err(_) => panic!(),
    }
}

const fn concat_buf(left: &'static str, right: &'static str) -> ([u8; 60], usize) {
    let mut buf = [b'7'; 32];
    let mut i = 0;
    while i < left.len() {
        buf[i] = left.as_bytes()[i];
        i += 1;
    }
    while i - left.len() < right.len() {
        buf[i] = right.as_bytes()[i - left.len()];
        i += 1;
    }

    (buf, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        struct DummyJob;
        impl Job for DummyJob {
            const NAME: &'static str = "dummy";
        }

        impl CronJob for DummyJob {
            const SCHEDULE: &'static str = "* * * * * *";
        }

        assert_eq!(DummyJob::CRON_NAME, "dummy-cron");
    }
}
