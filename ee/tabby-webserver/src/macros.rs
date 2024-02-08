#[macro_export]
macro_rules! enum_mapping {
    ($name:ty: $($variant:ident => $val:literal),+ $(,)?) => {
        impl tabby_db::conversions::DatabaseSerializable for $name {
            fn as_db_str(&self) -> &'static str {
                match self {
                    $(
                        <$name>::$variant => $val
                    ),+
                }
            }

            fn from_db_str(s: &str) -> anyhow::Result<Self> {
                match s {
                    $(
                        $val => Ok(<$name>::$variant),
                    )+
                    _ => Err(anyhow::anyhow!("{s} is not a valid {}", stringify!($name)))
                }
            }
        }
    };
}
