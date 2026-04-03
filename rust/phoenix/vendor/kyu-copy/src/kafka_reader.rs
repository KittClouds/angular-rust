//! Kafka reader stub for Windows/native builds that do not use Kafka COPY support.

use kyu_common::{KyuError, KyuResult};
use kyu_types::{LogicalType, TypedValue};

use crate::DataReader;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KafkaReader;

impl KafkaReader {
    pub fn open(_url: &str, _schema: &[LogicalType]) -> KyuResult<Self> {
        Err(KyuError::NotImplemented(
            "Kafka COPY support is not enabled in this patched build".into(),
        ))
    }
}

impl Iterator for KafkaReader {
    type Item = KyuResult<Vec<TypedValue>>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl DataReader for KafkaReader {
    fn schema(&self) -> &[LogicalType] {
        &[]
    }
}
