use compact_str::CompactString;
use phoenix_types::{BiTemporalWindow, TimeAnchorRecord};

#[derive(Clone, Debug, Default)]
pub struct TemporalBinding {
    pub anchor: Option<TimeAnchorRecord>,
    pub recorded_window: BiTemporalWindow,
}

pub struct TimeKernel;

impl TimeKernel {
    pub fn bind_label(label: &str, recorded_at: Option<i64>) -> TemporalBinding {
        TemporalBinding {
            anchor: Some(TimeAnchorRecord {
                time_id: None,
                label: CompactString::from(label),
                interval: BiTemporalWindow {
                    valid_from: None,
                    valid_to: None,
                    recorded_from: recorded_at,
                    recorded_to: None,
                },
            }),
            recorded_window: BiTemporalWindow {
                valid_from: None,
                valid_to: None,
                recorded_from: recorded_at,
                recorded_to: None,
            },
        }
    }
}
