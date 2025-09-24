use chrono::{DateTime, TimeZone, Utc};
pub use cron::Schedule;
use futures::Stream;

/// Represents a stream from a cron schedule with a timezone
#[derive(Clone, Debug)]
pub struct CronStream<Tz> {
    schedule: Schedule,
    timezone: Tz,
}

impl CronStream<Utc> {
    /// Build a new cron stream from a schedule using the UTC timezone
    pub fn new(schedule: Schedule) -> Self {
        Self {
            schedule,
            timezone: Utc,
        }
    }
}

impl<Tz> CronStream<Tz>
where
    Tz: TimeZone + Send + Sync + 'static,
    Tz::Offset: Send + Sync,
{
    /// Convert to consumable
    pub fn into_stream(self) -> impl Stream<Item = DateTime<Tz>> {
        let timezone = self.timezone.clone();
        let stream = async_stream::stream! {
            let mut schedule = self.schedule.upcoming_owned(timezone.clone());
            loop {
                let next = schedule.next();
                match next {
                    Some(next) => {
                        let to_sleep = next.clone() - timezone.from_utc_datetime(&Utc::now().naive_utc());
                        let to_sleep = match to_sleep.to_std() {
                            Ok(to_sleep) => to_sleep,
                            Err(_) => {
                                // If the next time is in the past or conversion fails, skip it and get the next one.
                                continue;
                            }
                        };
                        tokio::time::sleep(to_sleep).await;
                        yield next;
                    },
                    None => {
                        break;
                    }
                }
            }
        };
        Box::pin(stream)
    }
}
