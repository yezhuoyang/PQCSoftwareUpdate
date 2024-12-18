use ndarray::Array2;
use crate::number::inverse_mod;
use rand::Rng;

pub fn array2_to_vecvec<T: Clone>(array: Array2<T>) -> Vec<Vec<T>> {
    array
        .outer_iter() // Iterate over rows
        .map(|row| row.to_vec()) // Convert each row to a Vec
        .collect() // Collect rows into a Vec<Vec<T>>
}


pub fn matrix_vector_multiplication(A: &Vec<Vec<i64>>, x: &Vec<i64>, Q: &i64) -> Vec<i64> {
    let n = A.len();
    let m = A[0].len();
    if n == 0 || x.len()!= m {
        panic!("Matrix A and vector x must have compatible dimensions.");
    }
    let mut y = vec![0; n];
    for i in 0..n {
        for j in 0..m {
            y[i] = (y[i] + A[i][j] * x[j] % *Q) % *Q;
        }
    }
    y
}


pub fn matrix_matrix_multiplication(A: &Vec<Vec<i64>>, B: &Vec<Vec<i64>>, Q: &i64) -> Vec<Vec<i64>> {
    let n = A.len();
    let m = B[0].len();
    let l = B.len();
    if A[0].len() != l {
        panic!("Matrix dimensions are incompatible.");
    }
    let mut C = vec![vec![0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..l {
                C[i][j] = (C[i][j] + A[i][k] * B[k][j] % *Q) % *Q;
            }
        }
    }
    C
}



pub fn test_matrix_multiplication_example(){
    let A = vec![vec![1,1, 0, 0], vec![0, 1, 0, 3], vec![0, 0, 1, 2]];
    let s = vec![3,3, 4, 1];
    let y=matrix_vector_multiplication(&A, &s, &5);
    let result=vec![1, 1, 1];
    assert_eq!(y, result, "Matrix multiplication is incorrect");
}



pub fn row_rank_of_matrix(A: &Vec<Vec<i64>>, Q: &i64) -> usize {
    let n = A.len();
    let mut B = A.clone();
    let mut rank = 0;
    for i in 0..n {
        // Find the pivot
        let mut pivot = i;
        for j in i + 1..n {
            if B[j][i] % *Q > B[pivot][i] % *Q {
                pivot = j;
            }
        }

        // Swap rows in B
        B.swap(i, pivot);

        // Ensure the pivot element is invertible
        let pivot_value = B[i][i] % *Q;
        if pivot_value == 0 {
            continue;
        }

        let pivot_inv = inverse_mod(pivot_value, *Q);

        // Normalize the pivot row
        for j in i..n {
            B[i][j] = (B[i][j] * pivot_inv) % *Q;
        }

        // Eliminate the ith column for rows below
        for j in i + 1..n {
            let factor = B[j][i] % *Q;
            for k in i..n {
                B[j][k] = (B[j][k] - factor * B[i][k] % *Q + *Q) % *Q;
            }
        }
        rank += 1;
    }
    rank
}



pub fn matrix_multiply_float(A: &Vec<Vec<f64>>, B: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = A.len();
    let m = B[0].len();
    let l = B.len();
    if A[0].len() != l {
        panic!("Matrix dimensions are incompatible.");
    }
    let mut C = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..l {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    C
}

pub fn identity_matrix_float(n: usize) -> Vec<Vec<f64>> {
    let mut I = vec![vec![0.0; n]; n];
    for i in 0..n {
        I[i][i] = 1.0;
    }
    I
}


pub fn calculate_matrix_inverse(B: &Vec<Vec<f64>>)->Vec<Vec<f64>>{
    let n=B.len();
    let mut A=vec![vec![0.0; n]; n];
    for i in 0..n{
        for j in 0..n{
            A[i][j]=B[i][j];
        }
    }
    let mut inv=vec![vec![0.0; n]; n];
    for i in 0..n{
        inv[i][i]=1.0;
    }
    for i in 0..n{
        let mut pivot=i;
        for j in i+1..n{
            if A[j][i].abs()>A[pivot][i].abs(){
                pivot=j;
            }
        }
        if A[pivot][i]==0.0{
            panic!("Matrix is singular");
        }
        if pivot!=i{
            A.swap(i,pivot);
            inv.swap(i,pivot);
        }
        let pivot_inv=1.0/A[i][i];
        for j in 0..n{
            A[i][j]*=pivot_inv;
            inv[i][j]*=pivot_inv;
        }
        for j in 0..n{
            if j!=i{
                let factor=A[j][i];
                for k in 0..n{
                    A[j][k]-=factor*A[i][k];
                    inv[j][k]-=factor*inv[i][k];
                }
            }
        }
    }
    inv
}


pub fn test_calculate_matrix_inverse(){
//Generate random 3*3,4*4,5*5 matrices B
//Calculate the inverse Binverse using the function
//Compare B*Binverse with identity matrix
    for _ in 0..100{
        let n=rand::thread_rng().gen_range(3..6);
        let mut B=vec![vec![0.0; n]; n];
        for i in 0..n{
            for j in 0..n{
                B[i][j]=rand::thread_rng().gen_range(0.0..10.0);
            }
        }
        let Binverse=calculate_matrix_inverse(&B);
        let I=identity_matrix_float(n);
        let result=matrix_multiply_float(&B, &Binverse);
        println!("Matrix B*Binverse: {:?}", result);
        for i in 0..n{
            for j in 0..n{
                assert!((result[i][j]-I[i][j]).abs()<1e-6);
            }
        }
    }
}

pub fn nearest_integer_mod_q(x: f64, Q: i64) -> i64 {
    let x = x.round();
    let x = x as i64;
    x.rem_euclid(Q)
}


pub fn convert_int_vector_to_float_vector(A: &Vec<i64>) -> Vec <f64> {
    let n = A.len();
    let mut B = vec![0.0; n];
    for i in 0..n {
        B[i] = A[i] as f64;
    }
    B
}

pub fn convert_int_matrix_to_float_matrix(A: &Vec<Vec<i64>>) -> Vec<Vec<f64>> {
    let n = A.len();
    let m = A[0].len();
    let mut B = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            B[i][j] = A[i][j] as f64;
        }
    }
    B
}



pub fn test_nearest_integer_mod_q(){
    //Use some exmples
    let x1=-0.6;
    let Q1=5;
    let result1=nearest_integer_mod_q(x1, Q1);
    assert_eq!(result1, 4);
    let x2=1.8;
    let Q2=7;
    let result2=nearest_integer_mod_q(x2, Q2);
    assert_eq!(result2, 2);
}


//First, extend the field to any real field, then solve the linear equation 
//Then, round off the solution to the nearest integer
pub fn solve_closest_vector_by_rounding_off(B: &Vec<Vec<i64>>, inputy: &Vec<i64>, Q: &i64)-> Vec<i64>{
    //Calculate B inverse B^{-1}
    let floatB=convert_int_matrix_to_float_matrix(B);
    let floatBinv=calculate_matrix_inverse(&floatB);

    //Conver y to float vector
    let matrixinputy=vec![inputy.clone()];
    let floaty=convert_int_matrix_to_float_matrix(&matrixinputy);
    //Calculate floaty*B^{-1}
    let float_coeffs=matrix_multiply_float(&floaty, &floatBinv);
    //Round coeffs to the nearest integer
    let mut coeffs=vec![vec![0; float_coeffs[0].len()]];
    for i in 0..float_coeffs.len(){
        coeffs[0][i]=nearest_integer_mod_q(float_coeffs[0][i], Q.clone());
    }
    //Multiply B back
    let result=matrix_matrix_multiplication(&coeffs,B,Q);
    result[0].clone()
}


//return v1-v2
pub fn vector_delete(v1: &Vec<i64>, v2: &Vec<i64>, Q: &i64) -> Vec<i64> {
    let n = v1.len();
    if n != v2.len() {
        panic!("Vector dimensions are incompatible.");
    }
    let mut result = vec![0; n];
    for i in 0..n {
        result[i] = (v1[i] - v2[i] + *Q) % *Q;
    }
    result
}



pub fn test_solve_closest_vector_by_rounding_off(){
    let B=vec![vec![3,2,3,1], vec![0,4,1,2], vec![4,0,4,3], vec![2,3,2,3]];
    let rankB=row_rank_of_matrix(&B, &5);
    if rankB!=4{
        println!("Matrix B is singular, rank is {}", rankB);
        return;
    }
    let s=vec![3,3,4,1];
    let c0=solve_closest_vector_by_rounding_off(&B, &s, &5);
    println!("Closest vector: {:?}", c0);
}


pub fn solve_linear_by_gaussian_elimination(A: &Vec<Vec<i64>>, inputy: &Vec<i64>, Q: &i64) -> Vec<i64> {
    let n = A.len(); // Number of rows
    let m = A[0].len(); // Number of columns

    if n == 0 || inputy.len() != n {
        panic!("Matrix A and vector y must have compatible dimensions.");
    }

    let mut B = A.clone();
    let mut x = vec![0; m]; // Solution vector for size m
    let mut y = inputy.clone();

    // Gaussian elimination
    for i in 0..n {
        // Find the pivot
        let mut pivot = i;
        for j in i + 1..n {
            if B[j][i] % *Q > B[pivot][i] % *Q {
                pivot = j;
            }
        }

        // Swap rows in B and y
        B.swap(i, pivot);
        y.swap(i, pivot);

        // Ensure the pivot element is invertible
        let pivot_value = B[i][i] % *Q;
        if pivot_value == 0 {
            panic!("Matrix A is singular modulo {}", *Q);
        }

        let pivot_inv = inverse_mod(pivot_value, *Q);

        // Normalize the pivot row
        for j in i..m {
            B[i][j] = (B[i][j] * pivot_inv) % *Q;
        }
        y[i] = (y[i] * pivot_inv) % *Q;

        // Eliminate the ith column for all rows except the pivot row
        for j in 0..n {
            if j != i {
                let factor = B[j][i] % *Q;
                for k in i..m {
                    B[j][k] = (B[j][k] - factor * B[i][k] % *Q + *Q) % *Q;
                }
                y[j] = (y[j] - factor * y[i] % *Q + *Q) % *Q;
            }
        }
    }

    // Back substitution (extract solution from reduced matrix)
    for i in 0..n {
        x[i] = y[i];
    }

    // Return solution
    x
}






pub fn test_solve_linear_example(){
    //Generate a single example of a 3*3 matrix and a 3*1 vector
    //Solve the linear system using Gaussian elimination
    //Check the result
    let A = vec![vec![4, 0, 0, 9], vec![5, 5, 5, 4], vec![5, 2, 4, 2], vec![6, 6, 5, 0]];
    let y = vec![2, 1, 1, 4];
    let Q = 17;
    let x = solve_linear_by_gaussian_elimination(&A, &y, &Q);
    println!("Solution: {:?}", x);
    let result=matrix_vector_multiplication(&A, &x, &Q);
    println!("Result: {:?}", result);
    assert_eq!(y, result, "Solution is incorrect");
}


pub fn test_solve_linear_by_gaussian_elimination() {
    // Test the solve_linear_by_gaussian_elimination function
    // Randomly generated small 3*3,4*4,5*5 matrices, use brute force to calculate the determinant
    // Compare the results
    let Q = 23;
    for _ in 0..100 {
        let n = rand::thread_rng().gen_range(3..6);
        let mut A = vec![vec![0; n]; n];
        for i in 0..n {
            for j in 0..n {
                A[i][j] = rand::thread_rng().gen_range(0..10);
            }
        }
        let rowrank=row_rank_of_matrix(&A, &Q);
        if rowrank!=n{
            continue;
        }
        let mut y = vec![0; n];
        for i in 0..n {
            y[i] = rand::thread_rng().gen_range(0..10);
        }

        let x = solve_linear_by_gaussian_elimination(&A, &y, &Q);

        let result=matrix_vector_multiplication(&A, &x, &Q);

        println!("Matrix A: {:?}", A);
        println!("Vector y: {:?}", y);
        assert_eq!(y, result, "Solution is incorrect");
    }

    for _ in 0..100 {
        let n = rand::thread_rng().gen_range(5..10);
        let mut A = vec![vec![0; 2*n]; n];
        for i in 0..n {
            for j in 0..2*n {
                A[i][j] = rand::thread_rng().gen_range(0..15);
            }
        }
        let rowrank=row_rank_of_matrix(&A, &Q);
        if rowrank!=n{
            continue;
        }
        let mut y = vec![0; n];
        for i in 0..n {
            y[i] = rand::thread_rng().gen_range(0..15);
        }

        let x = solve_linear_by_gaussian_elimination(&A, &y, &Q);

        let result=matrix_vector_multiplication(&A, &x, &Q);

        println!("Matrix A: {:?}", A);
        println!("Vector y: {:?}", y);
        assert_eq!(y, result, "Solution is incorrect");
    }
}


//The extended gcd algorithm for integers. Input a,b
//Return s,t,gcd(a,b), such that sa+tb=gcd(a,b)
pub fn extended_gcd_integer(a: i64, b: i64) -> (i64, i64, i64) {
    let mut old_r = a;
    let mut r = b;
    let mut old_s = 1;
    let mut s = 0;
    let mut old_t = 0;
    let mut t = 1;
    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;

        let temp_t = t;
        t = old_t - quotient * t;
        old_t = temp_t;
    }
    (old_s, old_t, old_r)
}

pub fn extended_gcd_integer_test() {
    // Test the extended GCD algorithm for integers
    // Randomly generated two large numbers for 100 times
    for _ in 0..100 {
        let a = rand::thread_rng().gen_range(1..1000);
        let b = rand::thread_rng().gen_range(1..1000);
        let (s, t, gcd) = extended_gcd_integer(a, b);
        assert_eq!(s * a + t * b, gcd, "GCD is incorrect");
        println!("GCD of {} and {}: {}, s = {}, t = {}", a, b, gcd, s, t);
    }
}



//Use Gaussian elimination to calculate the determinant of an integer matrix
pub fn gaussian_elimination_determinant(mut matrix: Array2<i64>) -> i64 {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "Matrix must be square!");

    let mut det = 1; // To track the determinant
    for i in 0..n {
        // Find the pivot row
        let mut max_row = i;
        for k in (i + 1)..n {
            if matrix[[k, i]].abs() > matrix[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..n {
                matrix.swap((i, j), (max_row, j));
            }
            det *= -1; // Adjust determinant sign
        }

        // Check if the matrix is singular
        if matrix[[i, i]] == 0 {
            return 0;
        }

        // Eliminate below the pivot
        for k in (i + 1)..n {
            if matrix[[i, i]] == 0 {
                continue;
            }
            let factor = matrix[[k, i]] / matrix[[i, i]]; // Integer division
            for j in i..n {
                matrix[[k, j]] -= factor * matrix[[i, j]];
            }
        }

        // Multiply determinant by the diagonal element
        det *= matrix[[i, i]];
    }
    det
}


pub fn brute_force_determinant(matrix: Array2<i64>) -> i64 {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "Matrix must be square!");

    // Base case: 1x1 matrix
    if n == 1 {
        return matrix[[0, 0]];
    }

    // Base case: 2x2 matrix
    if n == 2 {
        return matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
    }

    // Recursive case: compute determinant using Laplace expansion
    let mut determinant = 0;
    for col in 0..n {
        // Get the minor matrix by excluding the current row and column
        let minor = get_minor(&matrix, 0, col);
        let sign = if col % 2 == 0 { 1 } else { -1 };
        determinant += sign * matrix[[0, col]] * brute_force_determinant(minor);
    }

    determinant
}




pub fn get_minor(matrix: &Array2<i64>, row_to_remove: usize, col_to_remove: usize) -> Array2<i64> {
    let n = matrix.nrows();
    let mut minor = Array2::zeros((n - 1, n - 1));

    let mut minor_row = 0;
    for row in 0..n {
        if row == row_to_remove {
            continue;
        }

        let mut minor_col = 0;
        for col in 0..n {
            if col == col_to_remove {
                continue;
            }

            minor[[minor_row, minor_col]] = matrix[[row, col]];
            minor_col += 1;
        }

        minor_row += 1;
    }

    minor
}


pub fn test_determinant_gaussian_elimination() {
    // Test the Gaussian elimination determinant algorithm
    // Randomly generated small 3*3,4*4,5*5 matrices, use brute force to calculate the determinant
    // Compare the results
    for _ in 0..100 {
        let n = rand::thread_rng().gen_range(3..6);
        let mut matrix = Array2::<i64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = rand::thread_rng().gen_range(0..10);
            }
        }
        let det = gaussian_elimination_determinant(matrix.clone());
        let det_brute = brute_force_determinant(matrix.clone());
        assert_eq!(det, det_brute, "Determinant is incorrect");
    } 
}


