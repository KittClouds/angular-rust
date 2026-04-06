use roaring::RoaringBitmap;

#[derive(Clone, Debug)]
pub enum PostingSet {
    Small(Vec<u32>),
    Large(RoaringBitmap),
}

impl Default for PostingSet {
    fn default() -> Self {
        Self::Small(Vec::new())
    }
}

impl PostingSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Small(values) => values.len(),
            Self::Large(bitmap) => bitmap.len() as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&mut self, ordinal: u32, threshold: usize) {
        match self {
            Self::Small(values) => {
                if values.last().map_or(true, |&last| ordinal > last) {
                    values.push(ordinal);
                } else if let Err(index) = values.binary_search(&ordinal) {
                    values.insert(index, ordinal);
                }

                if values.len() >= threshold {
                    let mut bitmap = RoaringBitmap::new();
                    for value in values.iter().copied() {
                        bitmap.insert(value);
                    }
                    *self = Self::Large(bitmap);
                }
            }
            Self::Large(bitmap) => {
                bitmap.insert(ordinal);
            }
        }
    }

    pub fn contains(&self, ordinal: u32) -> bool {
        match self {
            Self::Small(values) => values.binary_search(&ordinal).is_ok(),
            Self::Large(bitmap) => bitmap.contains(ordinal),
        }
    }

    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            Self::Small(values) => values.clone(),
            Self::Large(bitmap) => bitmap.iter().collect(),
        }
    }

    pub fn intersect(&self, other: &Self, threshold: usize) -> Self {
        match (self, other) {
            (Self::Small(left), Self::Small(right)) => {
                let mut result = Vec::with_capacity(left.len().min(right.len()));
                let mut left_index = 0usize;
                let mut right_index = 0usize;
                while left_index < left.len() && right_index < right.len() {
                    match left[left_index].cmp(&right[right_index]) {
                        std::cmp::Ordering::Less => left_index += 1,
                        std::cmp::Ordering::Greater => right_index += 1,
                        std::cmp::Ordering::Equal => {
                            result.push(left[left_index]);
                            left_index += 1;
                            right_index += 1;
                        }
                    }
                }
                Self::from_sorted(result, threshold)
            }
            (Self::Large(left), Self::Large(right)) => {
                let bitmap = left & right;
                Self::from_bitmap(bitmap, threshold)
            }
            (Self::Small(values), Self::Large(bitmap))
            | (Self::Large(bitmap), Self::Small(values)) => {
                let result = values
                    .iter()
                    .copied()
                    .filter(|value| bitmap.contains(*value))
                    .collect::<Vec<_>>();
                Self::from_sorted(result, threshold)
            }
        }
    }

    fn from_sorted(values: Vec<u32>, threshold: usize) -> Self {
        if values.len() >= threshold {
            let mut bitmap = RoaringBitmap::new();
            for value in values {
                bitmap.insert(value);
            }
            Self::Large(bitmap)
        } else {
            Self::Small(values)
        }
    }

    fn from_bitmap(bitmap: RoaringBitmap, threshold: usize) -> Self {
        if bitmap.len() as usize >= threshold {
            Self::Large(bitmap)
        } else {
            Self::Small(bitmap.iter().collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postings_promote_from_slice_to_bitmap() {
        let mut posting = PostingSet::new();
        for ordinal in 0..4 {
            posting.add(ordinal, 4);
        }
        assert!(matches!(posting, PostingSet::Large(_)));
    }

    #[test]
    fn intersection_keeps_selective_ordinals() {
        let mut left = PostingSet::new();
        let mut right = PostingSet::new();
        for value in [1, 2, 3, 5] {
            left.add(value, 8);
        }
        for value in [2, 4, 5, 6] {
            right.add(value, 8);
        }

        let intersection = left.intersect(&right, 8);
        assert_eq!(intersection.to_vec(), vec![2, 5]);
    }
}
