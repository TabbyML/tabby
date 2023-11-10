use kdam::{tqdm, Bar};

pub fn tqdm(total: usize) -> Bar {
    tqdm!(total = total, ncols = 40, force_refresh = true)
}
