use rand::seq::SliceRandom;
use rand::Rng;

pub fn shuffle<T>(vec: &mut [T]) {
    vec.shuffle(&mut rand::thread_rng());
}

pub fn split_data<T: Clone>(data: &[T], prob: f64) -> (Vec<T>, Vec<T>) {
    let mut data_copy = data.to_vec();
    shuffle(&mut data_copy);
    let cut = ((data_copy.len() as f64) * prob).round() as usize;

    return (data_copy[..cut].to_vec(), data_copy[cut..].to_vec());
}

pub fn train_test_split<'a, T, U>(
    xs: &'a [T],
    ys: &'a [U],
    test_pct: f64,
) -> (Vec<&'a T>, Vec<&'a T>, Vec<&'a U>, Vec<&'a U>) {
    let mut idxs = Vec::new();

    for i in 0..xs.len() {
        idxs.push(i);
    }

    let (train_idxs, test_idxs) = split_data(&idxs, 1.0 - test_pct);

    let mut x_train = Vec::new();
    for i in train_idxs.iter() {
        x_train.push(&xs[*i]);
    }

    let mut x_test = Vec::new();
    for i in test_idxs.iter() {
        x_test.push(&xs[*i]);
    }

    let mut y_train = Vec::new();
    for i in train_idxs.iter() {
        y_train.push(&ys[*i]);
    }

    let mut y_test = Vec::new();
    for i in test_idxs.iter() {
        y_test.push(&ys[*i]);
    }

    return (x_train, x_test, y_train, y_test);
}

pub fn test_train_test_split() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..1000 {
        xs.push(i);
        ys.push(2 * i);
    }

    let (x_train, x_test, y_train, y_test) = train_test_split(&xs, &ys, 0.25);

    assert!(
        x_train.len() == y_train.len(),
        "x_train.len() != y_train.len()"
    );
    assert!(x_train.len() == 750, "x_train.len() != 750");
    assert!(x_test.len() == y_test.len(), "x_test.len() != y_test.len()");
    assert!(x_test.len() == 250, "x_train.len() != 250");

    for i in 0..750 {
        assert!(
            *y_train[i] == 2 * x_train[i],
            "y_train: {} !== 2 * x_train {}",
            y_train[i],
            x_train[i]
        )
    }

    for i in 0..250 {
        assert!(
            *y_test[i] == 2 * x_test[i],
            "y_test: {} !== 2 * x_test {}",
            y_test[i],
            x_test[i]
        )
    }
}

pub fn accuracy(
    true_positive: i32,
    false_positive: i32,
    false_negative: i32,
    true_negative: i32,
) -> f64 {
    let correct = true_positive + true_negative;
    let total = true_positive + false_positive + false_negative + true_negative;
    return correct as f64 / total as f64;
}

pub fn precision(true_positive: i32, false_positive: i32) -> f64 {
    return (true_positive as f64) / (true_positive + false_positive) as f64;
}

pub fn recall(true_positive: i32, false_negative: i32) -> f64 {
    return (true_positive as f64) / (true_positive + false_negative) as f64;
}

pub fn f1_score(true_positive: i32, false_positive: i32, false_negative: i32) -> f64 {
    let p = precision(true_positive, false_positive);
    let r = recall(true_positive, false_negative);
    return 2. * p * r / (p + r);
}
