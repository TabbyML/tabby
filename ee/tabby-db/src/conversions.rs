pub trait DatabaseSerializable: Sized {
    fn as_db_str(&self) -> &'static str;
    fn from_db_str(s: &str) -> anyhow::Result<Self>;
}
