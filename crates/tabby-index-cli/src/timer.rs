use serde::Serialize;
use tantivy::time::Instant;

pub struct OpenTimer<'a> {
    name: &'static str,
    timer_tree: &'a mut TimerTree,
    start: Instant,
    depth: u32,
}

impl OpenTimer<'_> {
    /// Starts timing a new named subtask
    ///
    /// The timer is stopped automatically
    /// when the `OpenTimer` is dropped.
    pub fn open(&mut self, name: &'static str) -> OpenTimer<'_> {
        OpenTimer {
            name,
            timer_tree: self.timer_tree,
            start: Instant::now(),
            depth: self.depth + 1,
        }
    }
}

impl Drop for OpenTimer<'_> {
    fn drop(&mut self) {
        self.timer_tree.timings.push(Timing {
            name: self.name,
            duration: self.start.elapsed().whole_microseconds() as i64,
            depth: self.depth,
        });
    }
}

/// Timing recording
#[derive(Debug, Serialize)]
pub struct Timing {
    name: &'static str,
    duration: i64,
    depth: u32,
}

/// Timer tree
#[derive(Debug, Serialize, Default)]
pub struct TimerTree {
    timings: Vec<Timing>,
}

impl TimerTree {
    /// Returns the total time elapsed in microseconds
    pub fn total_time(&self) -> i64 {
        self.timings.last().unwrap().duration
    }

    /// Open a new named subtask
    pub fn open(&mut self, name: &'static str) -> OpenTimer<'_> {
        OpenTimer {
            name,
            timer_tree: self,
            start: Instant::now(),
            depth: 0,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_timer() {
        let mut timer_tree = TimerTree::default();
        {
            let mut a = timer_tree.open("a");
            {
                let mut ab = a.open("b");
                {
                    let _abc = ab.open("c");
                }
                {
                    let _abd = ab.open("d");
                }
            }
        }
        assert_eq!(timer_tree.timings.len(), 4);
    }
}
