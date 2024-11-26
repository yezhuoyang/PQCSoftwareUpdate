
mod poly; // Declare the module
use poly::Polynomial; // Bring the `Polynomial` struct into scope
mod falcon; // Declare the module
use falcon::*; // Import all public items from utils

use ndarray::arr2;
/*
fn main() {
    let file_path = "C:/Users/73747/Documents/GitHub/PQCSoftwareUpdate/text.bin"; // Replace with your file path
    let hash_output_size = 64; // Output size in bytes (e.g., 64 for 512-bit hash)

    match hash_file_with_shake256(file_path, hash_output_size) {
        Ok(hash) => {
            println!("SHAKE-256 Hash: {}", hash);
        }
        Err(e) => {
            eprintln!("Error hashing file: {}", e);
        }
    }
}
*/





/*
fn main() {
    // Example: Representing a polynomial as a matrix
    let phi = Polynomial::new(vec![1, 0, 0, 0, 1], 5); // φ = x^4 + 1
    let f = Polynomial::new(vec![0, 0, 1, 0, 1], 5); // f = x^2 + 1
    let matrix = f.to_matrix(&phi);

    println!("Matrix representation of f:");
    for row in matrix {
        println!("{:?}", row);
    }

    // Example: Generating NTRU lattice keys
    let f = Polynomial::new(vec![-55, 11, -23, -23, 47, 16, 13, 61], 12289);
    let g = Polynomial::new(vec![-25, -24, 30, -3, 36, -39, 6, 0], 12289);
    let phi = Polynomial::new(vec![1, 0, 0, 0, 0, 0, 0, 0, 1], 12289);
    let keys = NtruKeys::generate(f, g, phi, 12289);

    println!("Public key h: {:?}", keys.h);
}
*/

/*
fn main() {
    let degree = 20; // Degree of the polynomial
    let mean = 0.0; // Mean of the Gaussian distribution
    let std_dev = 3.0; // Standard deviation of the Gaussian distribution

    // Generate a polynomial
    let polynomial = generate_gaussian_polynomial(degree, mean, std_dev);

    println!("Generated polynomial coefficients: {:?}", polynomial);
}
*/


/*
fn main() {
    // Example: Representing a polynomial as a matrix
    let phi = Polynomial::new(vec![1, 0, 0, 0, 1], 5); // φ = x^4 + 1
    let f = Polynomial::new(vec![0, 0, 1, 0, 1], 5); // f = x^2 + 1
    let matrix = f.to_matrix(&phi);
}
*/

fn main(){

    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1], 12289); // φ = x^8 + 1
    let f=Polynomial::new(vec![-55,11,-23,-23,47,16,13,61],12289); //f
    let g=Polynomial::new(vec![-25,-24,30,-3,36,-39,6],12289); //g
    let F=Polynomial::new(vec![58,20,17,-64,-3,-9,-21,-84],12289); //G
    let G=Polynomial::new(vec![-41,-34,-33,25,-41,31,-18,-32],12289); //G
    let h=Polynomial::new(vec![-4839,-6036,-4459,-2665,-186,-4303,3388,-3568],12289); //h


    let mut ntrukeys=NtruKeys::generate_lattice(f,g,F,G,h,phi,12289);
    //println!("{ }",ntrukeys);

   

    let A: &[[i32; 4]] = &[
        [1, 1, 0,0],
        [0, 1, 0,3],
        [0, 0, 1,2],
    ];
    let Amat = arr2(A);
    let B: &[[i32; 4]] = &[
        [3, 2, 3, 1],
        [1, 4, 1, 2],
        [4, 1, 4, 3],
        [2, 3, 2, 4],
    ];
    let Bmat = arr2(B);

    let result=verify_lattice_orthorgonal(Amat,Bmat,5);

    println!("{:?}",result);
}