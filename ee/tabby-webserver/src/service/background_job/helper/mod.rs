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
        let name = format!("{}-cron", Self::NAME);
        WorkerBuilder::new(&name)
            .data(db.clone())
            .stream(stream)
            .layer(ConcurrencyLimitLayer::new(1))
            .layer(JobLogLayer::new(db, &name))
    }
}
