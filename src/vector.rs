use std::vec;

pub struct Vector {
    pub elements: Vec<f64>,
}

impl Vector {
    /*  pub fn new(&mut self, initial_vector: Vec<f64>) {
        self.elements = intial_vector
    }*/

    fn assert_equal_length(&self, w: Vec<f64>) {
        assert!(
            self.elements.len() == w.len(),
            "Elements must be of same length, got elements with length {} and {}",
            self.elements.len(),
            w.len()
        );
    }

    pub fn add(&self, w: &Vec<f64>) -> Vector {
        assert!(
            self.elements.len() == w.len(),
            "Elements must be of same length, got elements with length {} and {}",
            self.elements.len(),
            w.len()
        );

        let mut result: Vec<f64> = Vec::new();

        for i in 0..self.elements.len() {
            result.push(self.elements[i] + w[i])
        }

        return Vector { elements: result };
    }

    pub fn subtract(&self, w: &Vec<f64>) -> Vector {
        assert!(
            self.elements.len() == w.len(),
            "Elements must be of same length, got elements with length {} and {}",
            self.elements.len(),
            w.len()
        );

        let mut result: Vec<f64> = Vec::new();

        for i in 0..self.elements.len() {
            result.push(self.elements[i] - w[i])
        }

        return Vector { elements: result };
    }

    pub fn vectors_sum(&self, vectors: &Vec<&Vector>) -> Vector {
        let mut is_vectors_equal_length = true;

        for vector in vectors.iter() {
            if vector.elements.len() != self.elements.len() {
                is_vectors_equal_length = false;
            }
        }

        assert!(is_vectors_equal_length, "Not all vectors are equal length");

        let mut result: Vec<f64> = Vec::new();

        for i in 0..self.elements.len() {
            let mut sum = self.elements[i];
            for vector in vectors.iter() {
                sum += vector.elements[i];
            }
            result.push(sum);
        }

        return Vector { elements: result };
    }

    pub fn scalar_multiply(&self, c: f64) -> Vector {
        let mut result: Vec<f64> = Vec::new();

        for e in self.elements.iter() {
            result.push(e * c);
        }

        return Vector { elements: result };
    }

    pub fn dot(&self, w: &Vec<f64>) -> f64 {
        assert!(
            self.elements.len() == w.len(),
            "Elements must be of same length, got elements with length {} and {}",
            self.elements.len(),
            w.len()
        );

        let mut result: f64 = 0.0;

        for i in 0..self.elements.len() {
            result += self.elements[i] * w[i];
        }

        return result;
    }

    pub fn sum_of_squares(&self) -> f64 {
        self.dot(&self.elements)
    }

    pub fn magnitude(&self) -> f64 {
        self.sum_of_squares().sqrt()
    }

    pub fn squared_distance(&self, w: &Vec<f64>) -> f64 {
        self.subtract(w).sum_of_squares()
    }

    pub fn distance(&self, w: &Vec<f64>) -> f64 {
        self.squared_distance(w).sqrt()
    }
}

pub fn vectors_sum_(vectors: Vec<&Vector>) -> Vector {
    assert!(vectors.len() > 0, "Need at least one vector, yo");

    let mut is_vectors_equal_length = true;
    let n = vectors[0].elements.len();

    for vector in vectors.iter() {
        if vector.elements.len() != n {
            is_vectors_equal_length = false;
        }
    }

    assert!(is_vectors_equal_length, "Not all vectors are equal length");

    let mut result: Vec<f64> = Vec::new();

    for i in 0..n {
        let mut sum: f64 = 0.0;
        for vector in vectors.iter() {
            sum += vector.elements[i];
        }
        result.push(sum);
    }

    return Vector { elements: result };
}

pub fn vectors_sum(vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    assert!(vectors.len() > 0, "Need at least one vector, yo");

    let mut is_vectors_equal_length = true;
    let n = vectors[0].len();

    for vector in vectors.iter() {
        if vector.len() != n {
            is_vectors_equal_length = false;
        }
    }

    assert!(is_vectors_equal_length, "Not all vectors are equal length");

    let mut result: Vec<f64> = Vec::new();

    for i in 0..n {
        let mut sum: f64 = 0.0;
        for vector in vectors.iter() {
            sum += vector[i];
        }
        result.push(sum);
    }

    return result;
}

pub fn vector_mean_(vectors: Vec<&Vector>) -> Vector {
    let n = vectors.len();

    return Vector {
        elements: vectors_sum_(vectors)
            .scalar_multiply(1.0 / (n as f64))
            .elements,
    };
}

pub fn vector_mean(vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let n = vectors.len();
    return scalar_multiply(&vectors_sum(&vectors), 1. / (n as f64));
}

pub fn add(v: &[f64], w: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();

    for i in 0..v.len() {
        result.push(v[i] + w[i]);
    }

    return result;
}

pub fn scalar_multiply(v: &[f64], c: f64) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();

    for v_i in v.iter() {
        result.push(v_i * c)
    }

    return result;
}

pub fn dot(v: &[f64], w: &[f64]) -> f64 {
    assert!(
        v.len() == w.len(),
        "Elements must be of same length, got elements with length {} and {}",
        v.len(),
        w.len()
    );

    let mut result: f64 = 0.0;

    for i in 0..v.len() {
        result += v[i] * w[i];
    }

    return result;
}

pub fn sum_of_squares(v: &[f64]) -> f64 {
    dot(v, v)
}

pub fn magnitude(v: &[f64]) -> f64 {
    sum_of_squares(v).sqrt()
}

pub fn subtract(v: &[f64], w: &[f64]) -> Vector {
    assert!(
        v.len() == w.len(),
        "Elements must be of same length, got elements with length {} and {}",
        v.len(),
        w.len()
    );

    let mut result: Vec<f64> = Vec::new();

    for i in 0..v.len() {
        result.push(v[i] - w[i])
    }

    return Vector { elements: result };
}

pub fn squared_distance(v: &[f64], w: &[f64]) -> f64 {
    subtract(v, w).sum_of_squares()
}

pub fn distance(v: &[f64], w: &[f64]) -> f64 {
    squared_distance(v, w).sqrt()
}
