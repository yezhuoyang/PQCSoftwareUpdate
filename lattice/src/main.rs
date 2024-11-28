
mod poly; // Declare the module
use poly::*; // Bring the `Polynomial` struct into scope
mod falcon; // Declare the module
use falcon::*; // Import all public items from utils
use ndarray::arr2;

mod config;
use config::*; // Import all constants into the local scope
use num::complex::Complex;

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
    let pairvec=calculate_unit_roots(&8);
    let phi = Polynomial::new(vec![1, 0, 0, 0, 0, 0, 0, 0, 1]); // φ = x^8 + 1
    let poly=Polynomial::new(vec![-55,11,-23,-23,47,16,13]); //f
    
    let  K=FFT(&poly, &phi);
    println!("{:?}",K);


}