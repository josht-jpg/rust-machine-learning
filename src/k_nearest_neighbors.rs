use crate::{machine_learning::split_data, vector};
use std::collections::HashMap;
extern crate csv;

extern crate serde;

extern crate rustlearn;
use rustlearn::datasets::iris;

// TODO: this is pretty hairy
pub fn raw_majority_votes(labels: &[char]) -> char {
    let mut labels_ = HashMap::new();
    for label in labels {
        if labels_.contains_key(label) {
            *labels_.get_mut(label).unwrap() += 1;
        } else {
            labels_.insert(label, 1);
        }
    }

    let mut highest_count = 0;
    let mut label_with_highest_count = labels[0];
    for (key, value) in labels_ {
        if value > highest_count {
            highest_count = value;
            label_with_highest_count = *key;
        }
    }

    return label_with_highest_count;
}

pub fn majority_vote_(labels: &[char]) -> char {
    let mut labels_ = HashMap::new();
    for label in labels {
        if labels_.contains_key(label) {
            *labels_.get_mut(label).unwrap() += 1;
        } else {
            labels_.insert(label, 1);
        }
    }

    let mut highest_count = 0;
    let mut label_with_highest_count = labels[0];
    for (key, value) in &labels_ {
        if *value > highest_count {
            highest_count = *value;
            label_with_highest_count = **key;
        }
    }

    let mut num_winners = 0;
    for (key, value) in labels_ {
        if value == highest_count {
            num_winners += 1;
        }
    }

    if num_winners == 1 {
        return label_with_highest_count;
    } else {
        return majority_vote_(&labels[..labels.len() - 1]);
    }
}

pub fn majority_vote(labels: &[i32]) -> i32 {
    let mut labels_ = HashMap::new();
    for label in labels {
        if labels_.contains_key(label) {
            *labels_.get_mut(label).unwrap() += 1;
        } else {
            labels_.insert(label, 1);
        }
    }

    let mut highest_count = 0;
    let mut label_with_highest_count = labels[0];
    for (key, value) in &labels_ {
        if *value > highest_count {
            highest_count = *value;
            label_with_highest_count = **key;
        }
    }

    let mut num_winners = 0;
    for (key, value) in labels_ {
        if value == highest_count {
            num_winners += 1;
        }
    }

    if num_winners == 1 {
        return label_with_highest_count;
    } else {
        return majority_vote(&labels[..labels.len() - 1]);
    }
}

//Label?
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Labels {
    Setosa = 0,
    Versicolor = 1,
    Virginica = 2,
}

// For iris dataset

#[derive(Debug, Clone, Copy)]
//Maybe not a tuple
pub struct LabeledPoint {
    point: (f64, f64, f64, f64),
    //label: Labels,
    label: i32,
}

pub fn knn_classify_(k: i32, labeled_points: &mut [LabeledPoint], new_point: &[f64]) -> i32 {
    // Order the labeled points from nearest to farthest
    // TODO: maybe not in place?
    labeled_points.sort_by(|a, b| {
        vector::distance(&vec![a.point.0, a.point.1, a.point.2, a.point.3], new_point)
            .partial_cmp(&vector::distance(
                &vec![a.point.0, b.point.1, b.point.2, b.point.3],
                new_point,
            ))
            .unwrap()
    });

    let mut k_nearest_labels = Vec::new();
    for lp in &labeled_points[..(k as usize)] {
        k_nearest_labels.push(lp.label);
    }

    return majority_vote(&k_nearest_labels);
}

pub fn knn_classify(
    k: i32,
    labeled_points: &mut [LabeledPoint],
    new_point: (f64, f64, f64, f64),
) -> Labels {
    // Order the labeled points from nearest to farthest
    // TODO: maybe not in place?
    labeled_points.sort_by(|a, b| {
        vector::distance(
            &vec![a.point.0, a.point.1, a.point.2, a.point.3],
            &vec![new_point.0, new_point.1, new_point.2, new_point.3],
        )
        .partial_cmp(&vector::distance(
            &vec![a.point.0, b.point.1, b.point.2, b.point.3],
            &vec![new_point.0, new_point.1, new_point.2, new_point.3],
        ))
        .unwrap()
    });

    let mut k_nearest_labels = Vec::new();
    for lp in &labeled_points[..(k as usize)] {
        k_nearest_labels.push(lp.label);
    }

    // Could be wronng here
    match majority_vote(&k_nearest_labels) {
        0 => Labels::Setosa,
        1 => Labels::Versicolor,
        _ => Labels::Virginica,
    }

    // return majority_vote(&k_nearest_labels);
}

/*use serde::Deserialize;
#[derive(Debug, Deserialize)]
pub struct Record {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: String,
}

use std::error::Error;
use std::io;
use std::process;*/

pub fn load_iris() {
    let (arr1, arr2) = iris::load_data();
    println!("{:?}", arr2)
}

pub fn parse_iris_row(row: (f64, f64, f64, f64, i32)) -> LabeledPoint {
    //TODO: clean up
    let (sepal_length, sepal_width, petal_length, petal_width, label) = row;

    /*let mut label: Labels;
    if row.4 == 0 {
        label = Labels::Setosa;
    } else if row.4 == 1 {
        label = Labels::Versicolor;
    } else {
        label = Labels::Virginica;
    }*/

    return LabeledPoint {
        point: (sepal_length, sepal_width, petal_length, petal_width),
        label,
    };
}

pub fn kmeans_iris() {
    let (measurements, labels) = iris::load_data();

    //let mut data = Vec::new();

    //let data = measurements.data().iter().zip(labels.data().iter()); //.map(|(sepal_length, sepal_width, petal_length, petal_width, label)| )

    let mut sepal_length = 0.;
    let mut sepal_width = 0.;
    let mut petal_length = 0.;
    let mut petal_width = 0.;

    let mut data: Vec<LabeledPoint> = Vec::new();

    for i in 0..measurements.data().len() {
        // println!("{:?}", measurements.data()[i as usize]);

        match i % 4 {
            0 => sepal_length = measurements.data()[i as usize],
            1 => sepal_width = measurements.data()[i as usize],
            2 => petal_length = measurements.data()[i as usize],
            3 => {
                petal_width = measurements.data()[i as usize];

                /*println!(
                    "Measures: {} {} {} {}, label: {}",
                    sepal_length as f64,
                    sepal_width as f64,
                    petal_length as f64,
                    petal_width as f64,
                    labels.data()[((i + 1) / 4) - 1 as usize] as i32
                );*/

                data.push(LabeledPoint {
                    point: (
                        sepal_length as f64,
                        sepal_width as f64,
                        petal_length as f64,
                        petal_width as f64,
                    ),
                    label: labels.data()[((i + 1) / 4) - 1 as usize] as i32,
                })
            }
            _ => println!("Won't happen, yo"),
        }
    }
    let (mut iris_train, iris_test) = split_data(&data, 0.70);

    /*println!("{:?}", data);

    println!("{:?}", iris_train);
    println!("{:?}", iris_test);*/

    assert_eq!(iris_train.len(), 105);
    assert_eq!(iris_test.len(), 45);

    // track how many times we see (predicted, actual)

    let mut confusion_matrix: HashMap<(Labels, Labels), i32> = HashMap::new();
    let mut num_correct = 0;

    for iris in iris_test.iter() {
        let predicted = knn_classify(5, &mut iris_train, iris.point);
        let actual = match iris.label {
            0 => Labels::Setosa,
            1 => Labels::Versicolor,
            _ => Labels::Virginica,
        };

        // return majority_vote(&k_nearest_labels);

        if predicted == actual {
            num_correct += 1;
        }

        if confusion_matrix.contains_key(&(predicted, actual)) {
            *confusion_matrix.get_mut(&(predicted, actual)).unwrap() += 1;
        } else {
            confusion_matrix.insert((predicted, actual), 1);
        }

        //confusion_matrix[()]
    }

    let pct_correct = num_correct as f64 / iris_test.len() as f64;

    println!("{},  {:?}", pct_correct, confusion_matrix)
}
