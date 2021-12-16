use crate::vector::{self, vector_mean};
extern crate rand;

use rand::seq::SliceRandom;
use rand::Rng;

pub fn difference_quotient(f: fn(x: f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x)) / h
}

pub fn partial_difference_quotient(
    f: fn(v: &Vec<f64>) -> f64,
    v: &Vec<f64>,
    i: usize,
    h: f64,
) -> f64 {
    let mut w: Vec<f64> = Vec::new();

    for j in 0..v.len() {
        let mut add = 0.;

        if i == j {
            add = h;
        }

        w.push(v[i] + add);
    }

    return (f(&w) - f(&v)) / h;
}

pub fn estimate_gradient(f: fn(v: &Vec<f64>) -> f64, v: &Vec<f64>, i: i32, h: f64) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();

    for i in 0..v.len() {
        result.push(partial_difference_quotient(f, v, i, h))
    }

    return result;
}

pub fn gradient_step(v: &[f64], gradient: &Vec<f64>, step_size: f64) -> Vec<f64> {
    assert!(
        v.len() == gradient.len(),
        "v and gradient must be same length"
    );

    let step = vector::scalar_multiply(&gradient, step_size);

    return vector::add(&v, &step);
}

pub fn sum_of_squares_gradient(v: &[f64]) -> Vec<f64> {
    vector::scalar_multiply(&v, 2.)
}

pub fn linear_gradient(x: f64, y: f64, theta: &[f64]) -> Vec<f64> {
    let slope = theta[0];
    let intercept = theta[1];

    let predicted = slope * x + intercept;
    let error = predicted - y;

    let squared_error = error.powi(2);

    let grad = vec![2. * error * x, 2. * error];
    return grad;
}

pub fn run() {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f64> = (0..3)
        .map(|_| {
            // 1 (inclusive) to 21 (exclusive)
            println!("here");
            rng.gen_range::<i32, _>(-10..10) as f64
        })
        .collect();

    println!("{:?}:    ", v);

    for epoch in 0..1000 {
        let grad = sum_of_squares_gradient(&v);
        v = gradient_step(&v, &grad, -0.01);
        // println!("{}, [{:?}]\n\n\n\n\n\n\n ", epoch, v)
    }

    let dist = vector::distance(&v, &vec![0., 0., 0.]);

    println!("dist: {:?}, v: {:?}", dist, v);

    assert!(dist < 0.001, "dist too big")
}

pub fn example() {
    let mut inputs: Vec<[i32; 2]> = Vec::new();

    for x in -50..50 {
        inputs.push([x, 20 * x + 5]);
    }

    let mut rng = rand::thread_rng();
    let mut theta = vec![rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];

    let learning_rate = 0.001;

    for epoch in 0..5000 {
        let mut gradients = Vec::new();

        for [x, y] in inputs.iter() {
            let value = linear_gradient(*x as f64, *y as f64, &theta);
            gradients.push(value);
        }

        let grad = vector::vector_mean(&gradients);

        theta = gradient_step(&theta, &grad, -learning_rate);

        println!("{}, {:?}", epoch, theta);
    }

    assert!(19.9 < theta[0] && theta[0] < 20.1, "bad slope");
    assert!(4.9 < theta[1] && theta[1] < 5.1, "bad intercept");
}

pub fn minibatches<T>(dataset: &[T], batch_size: i32, shuffle: bool) -> Vec<&[T]> {
    let mut batch_starts = Vec::new();

    for start in (0..dataset.len()).step_by(batch_size as usize) {
        batch_starts.push(start);
    }

    if shuffle {
        batch_starts.shuffle(&mut rand::thread_rng());
    }

    let mut result = Vec::new();

    for start in batch_starts {
        let end = start + batch_size as usize;
        result.push(&dataset[start..end]);
    }

    return result;
}

pub fn minibatch_example() {
    let learning_rate = 0.001;

    let mut rng = rand::thread_rng();
    let mut theta = vec![rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];

    let mut inputs: Vec<[i32; 2]> = Vec::new();

    for x in -50..50 {
        inputs.push([x, 20 * x + 5]);
    }

    for epoch in 0..1000 {
        for batch in minibatches(&inputs, 20, true) {
            let mut vectors = Vec::new();

            for [x, y] in batch {
                vectors.push(linear_gradient(*x as f64, *y as f64, &theta));
                //vectors.push(x, y, theta);
            }

            let grad = vector::vector_mean(&vectors);
            theta = gradient_step(&theta, &grad, -learning_rate)
        }
    }

    println!("{}      {}", theta[0], theta[1]);

    assert!(
        19.9 < theta[0] && theta[0] < 20.1,
        "Slope should be about 20"
    );
    assert!(
        4.9 < theta[1] && theta[1] < 5.1,
        "Intercept should be about 5"
    );
}
