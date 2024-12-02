
mod poly; // Declare the module
use poly::*; // Bring the `Polynomial` struct into scope
mod falcon; // Declare the module
use falcon::*; // Import all public items from utils
use ndarray::arr2;

mod config;
use config::*; // Import all constants into the local scope
use num::complex::Complex;
use std::fs::File;

use std::fs;
use std::io;

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


fn read_string_from_file(file_path: &str) -> Result<String, io::Error> {
    fs::read_to_string(file_path)
}



/*
fn main(){
    let file_path = "C:/Users/73747/Documents/GitHub/PQCSoftwareUpdate/text.bin"; // Replace with your file path

    let content = read_string_from_file(file_path);

    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1]); // φ = x^8 + 1
    
    //Generate a pair of falcon public and private keys
    let falconkeys=NtruKeys::NTRUGen(phi);
    
    let signature=falconkeys.sign(content.unwrap());
    let verification=falconkeys.verify(content.unwrap(),signature);
    if verification{
        println!("The signature is valid");
    }
    else{
        println!("The signature is invalid");
    }

}
*/

/*
fn main() {

    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1]); // φ = x^8 + 1

    let f=Polynomial::new(vec![-55,11,-23,-23,47,16,13,61]); //f
    let g=Polynomial::new(vec![-25,-24,30,-3,36,-39,6]); //g


    //f*G-g*F=q mod phi
    let F=Polynomial::new(vec![58,20,17,-64,-3,-9,-21,-84]); //G
    let G=Polynomial::new(vec![-41,-34,-33,25,-41,31,-18,-32]); //G

    let h=Polynomial::new(vec![-4839,-6036,-4459,-2665,-186,-4303,3388,-3568]); //h


    let B=calculate_secret_key(&f,&g,&G,&F,&phi);
    let A=calculate_public_key(&h,&phi);

    println!("The public key is {:?}",A);
    println!("The secret key is {:?}",B);
}
*/

/*
fn main() {
    // Example: Representing a polynomial as a matrix
    let phi = Polynomial::new(vec![1, 0, 0, 0, 0 ,0 ,0 , 1]); // φ = x^8 + 1
    let poly = Polynomial::new(vec![2, 3, 1, 2, 1]); // f = x^2 + 1
    println!("The poly is {}",poly);
    let fft=FFT(&poly, &phi);
    println!("The FFT of the polynomial is {:?}",fft);

    let polyinverse=inverseFFT(&phi,&fft);
    println!("The Inverse FFT of the polynomial is {}",polyinverse);
}
*/




fn main() {
    //test_poly_deletion();
    //test_poly_equal();
    //test_poly_sum();
    //test_poly_multiplication();
    //extended_gcd_poly_test();
    //extended_gcd_poly_example();
    test_leading_coeff()
}
