use crate::vector;

pub struct Matrix {
    pub rows: Vec<vector::Vector>,
}

impl Matrix {
    pub fn shape(&self) -> (usize, usize) {
        if self.rows.len() == 0 {
            return (0, 0);
        }
        return (self.rows.len(), self.rows[0].elements.len());
    }

    pub fn row(&self, i: usize) -> &vector::Vector {
        &self.rows[i]
    }

    pub fn column(&self, i: usize) -> vector::Vector {
        //assert!

        let mut result: Vec<f64> = Vec::new();

        for v in self.rows.iter() {
            result.push(v.elements[i])
        }

        return vector::Vector { elements: result };
    }

    pub fn printMatrix(&self) {
        for row in self.rows.iter() {
            for i in 0..row.elements.len() {
                if i < row.elements.len() - 1 {
                    print!("{}, ", row.elements[i]);
                } else {
                    print!("{}", row.elements[i]);
                }
            }
            print!("\n");
        }
    }
}

pub fn make_matrix(
    num_rows: usize,
    num_cols: usize,
    entries_function: fn(i: usize, j: usize) -> f64,
) -> Matrix {
    let mut rows: Vec<vector::Vector> = Vec::new();

    for i in 0..num_rows {
        let mut column: Vec<f64> = Vec::new();
        for j in 0..num_cols {
            column.push(entries_function(i, j));
        }
        rows.push(vector::Vector { elements: column });
    }

    return Matrix { rows };
}

pub fn identity(n: usize) -> Matrix {
    make_matrix(n, n, |i, j| if i == j { return 1.0 } else { return 0.0 })
}
