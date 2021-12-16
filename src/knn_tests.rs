#[cfg(test)]
mod tests {
    use crate::k_nearest_neighbors;

    #[test]
    pub fn raw_majority_votes() {
        assert_eq!(
            k_nearest_neighbors::raw_majority_votes(&vec!['a', 'b', 'c', 'b', 'c', 'c']),
            'c'
        )
    }

    pub fn majority_votes() {
        assert_eq!(
            k_nearest_neighbors::raw_majority_votes(&vec!['a', 'b', 'c', 'b', 'a']),
            'b'
        )
    }
}
