mod gradient_descent;
mod k_nearest_neighbors;
mod knn_tests;
mod machine_learning;
mod machine_learning_tests;
mod matrix;
mod vector;

fn main() {
    /*  let v1 = vector::Vector {
        elements: vec![1, 2, 3],
    };

    let v2 = vector::Vector {
        elements: vec![3, 3, 3],
    };

    let sum = v1.add(&v2.elements);
    let diff = v1.subtract(&v2.elements);
    println!("{:?}", sum.elements);
    println!("{:?}", diff.elements);

    let v3 = vector::Vector {
        elements: vec![5, 1, 5],
    };

    let vectors_sum = v1.vectors_sum(&vec![&v2, &v3]);
    println!("{:?}", vectors_sum.elements);

    */

    /*

    let v1 = vector::Vector {
        elements: vec![1.0, 2.0],
    };

    let v2 = vector::Vector {
        elements: vec![3.0, 4.0],
    };

    let v3 = vector::Vector {
        elements: vec![5.0, 6.0],
    };

    println!("{:?}", vector::vector_mean(vec![&v1, &v2, &v3]).elements)

    */

    /*   let v1 = vector::Vector {
        elements: vec![1., 2., 3.],
    };

    let v2 = vector::Vector {
        elements: vec![4., 5., 6.],
    };

    println!("{}", v1.dot(&v2.elements));
    println!("{}", v1.sum_of_squares());

    let v3 = vector::Vector {
        elements: vec![3., 4.],
    };
    println!("{}", v3.magnitude()); */

    /*let A = matrix::Matrix {
        rows: vec![
            vector::Vector {
                elements: vec![1., 2., 3.],
            },
            vector::Vector {
                elements: vec![1., 2., 3.],
            },
        ],
    };
    println!("{:?}", A.shape());*/

    /*let I = matrix::identity(5);
    I.printMatrix();

    println!("\n{:?}", I.column(0).elements);

    let v1 = vector::Vector {
        elements: vec![1., 3.],
    };
    let v2 = vector::Vector {
        elements: vec![2., 4.],
    };

    let A = matrix::Matrix { rows: vec![v1, v2] };
    println!("\n{:?}", A.column(0).elements);*/

    // gradient_descent::example();

    // gradient_descent::minibatch_example();

    // machine_learning::test_train_test_split()

    k_nearest_neighbors::kmeans_iris();
}
