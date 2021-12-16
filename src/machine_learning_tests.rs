#[cfg(test)]
mod tests {
    use crate::machine_learning;

    #[test]
    pub fn accuracy() {
        assert_eq!(machine_learning::accuracy(70, 4930, 13930, 981070), 0.98114)
    }

    #[test]
    pub fn precision() {
        assert_eq!(machine_learning::precision(70, 4930), 0.014)
    }
}
