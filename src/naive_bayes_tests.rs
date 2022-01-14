#[cfg(test)]
mod tests {
    use crate::naive_bayes::{self, new_classifier, Counts, Message, NaiveBayesClassifier};
    use std::collections::{HashMap, HashSet};

    /* fn messages() -> [Message<'static>; 3] {
        return [
            Message {
                text: "spam rules",
                is_spam: true,
            },
            Message {
                text: "ham rules",
                is_spam: false,
            },
            Message {
                text: "hello ham",
                is_spam: false,
            },
        ];
    }*/

    /* pub fn tokenize() {
        let text = "I love Rust and Rust loves me".to_lowercase();
        let input = text.as_str();

        let result = naive_bayes::tokenize(input);

        println!("{:?}", result);

        let mut expected: HashSet<&str> = HashSet::new();
        expected.insert("i");
        expected.insert("love");
        expected.insert("rust");
        expected.insert("and");
        expected.insert("loves");
        expected.insert("me");
        // expected.insert("<3");

        assert_eq!(result, expected)
    } */

    #[test]
    fn naive_bayes_tests() {
        let messages = [
            Message {
                text: "spam rules",
                is_spam: true,
            },
            Message {
                text: "ham rules",
                is_spam: false,
            },
            Message {
                text: "hello ham",
                is_spam: false,
            },
        ];

        let mut model = new_classifier(0.5);
        model.train(&messages);

        let mut expected_tokens: HashMap<String, Counts> = HashMap::from([
            ("spam".to_string(), Counts { spam: 1, ham: 0 }),
            ("rules".to_string(), Counts { spam: 1, ham: 1 }),
            ("ham".to_string(), Counts { spam: 0, ham: 2 }),
            ("hello".to_string(), Counts { spam: 0, ham: 1 }),
        ]);

        assert_eq!(model.tokens, expected_tokens);
        assert_eq!(model.totals.spam, 1);
        assert_eq!(model.totals.ham, 2);

        let text = "hello spam";

        let probs_if_spam = [
            (1. + 0.5) / (1. + 2. * 0.5),      // "spam"  (present)
            1. - (0. + 0.5) / (1. + 2. * 0.5), // "ham"   (not present)
            1. - (1. + 0.5) / (1. + 2. * 0.5), // "rules" (not present)
            (0. + 0.5) / (1. + 2. * 0.5),      // "hello" (present)
        ];

        let probs_if_ham = [
            (0. + 0.5) / (2. + 2. * 0.5),      // "spam"  (present)
            1. - (2. + 0.5) / (2. + 2. * 0.5), // "ham"   (not present)
            1. - (1. + 0.5) / (2. + 2. * 0.5), // "rules" (not present)
            (1. + 0.5) / (2. + 2. * 0.5),      // "hello" (present)
        ];

        let p_if_spam_exp: f64 = probs_if_spam.iter().map(|p| (*p as f64).ln()).sum();
        let p_if_spam = p_if_spam_exp.exp();

        let p_if_ham_exp: f64 = probs_if_ham.iter().map(|p| (*p as f64).ln()).sum();
        let p_if_ham = p_if_ham_exp.exp();

        assert_eq!(model.predict(text), p_if_spam / (p_if_spam + p_if_ham))
    }
}
