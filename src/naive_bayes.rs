extern crate lazy_static;
extern crate regex;

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::{HashMap, HashSet};

//use crat::utils::e;

//has to be lower case, maybe fix that
pub fn tokenize(text: &str) -> HashSet<&str> {
    // let lower_case_text = text.to_lowercase();

    // https://rust-lang-nursery.github.io/rust-cookbook/text/regex.html
    lazy_static! {
        static ref HASHTAG_REGEX: Regex = Regex::new(r"[a-z0-9']+").unwrap();
    }
    HASHTAG_REGEX
        .find_iter(text)
        .map(|mat| mat.as_str())
        .collect()
}

#[derive(Debug)]
pub struct Message<'a> {
    pub text: &'a str,
    pub is_spam: bool,
}

pub struct NaiveBayesClassifier {
    k: f64,
    pub tokens: HashMap<String, Counts>,
    pub totals: Counts,
}

#[derive(PartialEq, Eq, Default, Debug)]
pub struct Counts {
    pub spam: u32,
    pub ham: u32,
}

pub fn new_classifier(k: f64) -> NaiveBayesClassifier {
    return NaiveBayesClassifier {
        k,
        tokens: HashMap::new(),
        totals: Counts { ham: 0, spam: 0 },
    };
}

impl NaiveBayesClassifier {
    pub fn train(&mut self, messages: &[Message]) {
        for i in 0..messages.len() {
            let message = &messages[i];
            if message.is_spam {
                self.totals.spam += 1;
            } else {
                self.totals.ham += 1;
            }

            // Increment word counts
            for token in tokenize(message.text) {
                let counts = self.tokens.entry(token.to_string()).or_default();

                if message.is_spam {
                    counts.spam += 1;
                } else {
                    counts.ham += 1;
                }
            }
        }
    }

    fn probabilites(&self, token: &str) -> (f64, f64) {
        println!("token: {}, hasmap: {:?}", token, self.tokens);

        let Counts { ham, spam } = self.tokens[token];

        let prob_of_token_spam = (spam as f64 + self.k) / (self.totals.spam as f64 + 2. * self.k);
        let prob_of_token_ham = (ham as f64 + self.k) / (self.totals.ham as f64 + 2. * self.k);

        return (prob_of_token_spam, prob_of_token_ham);
    }

    pub fn predict(&self, text: &str) -> f64 {
        let text_tokens = tokenize(text);
        let mut log_prob_if_spam = 0.;
        let mut log_prob_if_ham = 0.;

        for token in self.tokens.keys() {
            let (prob_if_spam, prob_if_ham) = self.probabilites(&token);

            // If *token* appears in the message,
            // add the log probability of seeing it
            if text_tokens.contains(token.as_str()) {
                log_prob_if_spam += prob_if_spam.ln();
                log_prob_if_ham += prob_if_ham.ln();
            } else {
                log_prob_if_spam += (1. - prob_if_spam).ln();
                log_prob_if_ham += (1. - prob_if_ham).ln();
            }
        }

        let prob_if_spam = log_prob_if_spam.exp();
        let prob_if_ham = log_prob_if_ham.exp();

        return prob_if_spam / (prob_if_spam + prob_if_ham);
    }
}

/*impl Default for NaiveBayesClassifier {
    fn default() -> NaiveBayesClassifier {
        spam_messages: 0,
        ham_messages: 0,
    }
}*/
