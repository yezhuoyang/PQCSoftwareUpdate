
mod poly; // Declare the module
use poly::*; // Bring the `Polynomial` struct into scope
mod falcon; // Declare the module
use falcon::*; // Import all public items from utils
use ndarray::arr2;

mod config;
use config::*; // Import all constants into the local scope

mod number; // Declare the module
use number::*; // Import all public items from utils
mod linearAlg; // Declare the module
use linearAlg::*; // Import all public items from utils

use num::complex::Complex;
use std::fs::File;

use std::fs;
use std::io;



use std::io::{Write, BufWriter};
use std::time::Instant;


fn read_string_from_file(file_path: &str) -> Result<String, io::Error> {
    fs::read_to_string(file_path)
}



fn main() {
    let beta = 1;

    // Test with varying degrees of φ
    for degree in [8, 16, 32, 64, 128].iter() {
        // Generate φ = x^degree + 1
        let mut phi_coeffs = vec![0; *degree];
        phi_coeffs.push(1); // Add 1 at the end for x^degree
        phi_coeffs[0] = 1;  // Add 1 at the start for the constant term
        let phi = Polynomial::new(phi_coeffs);

        // Start timer
        let start_time = Instant::now();

        // Generate NTRU keys
        let ntrukeys = NtruKeys::NTRUGen(&phi);

        // Sign a message
        let message = "UCLA is the #1 Public School".to_string();
        let signature = ntrukeys.sign(message, beta);

        // Measure elapsed time
        let time_taken = start_time.elapsed().as_millis();

        // Print the degree and time taken
        println!("Degree: {}, Time Taken: {} ms", degree, time_taken);
    }
}

/*
fn main() {
    /*
    let beta = 1;
    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1]); // φ = x^8 + 1
    let f=Polynomial::new(vec![-55,11,-23,-23,47,16,13,61]); //f
    let g=Polynomial::new(vec![-25,-24,30,-3,36,-39,6]); //g
    let F=Polynomial::new(vec![58,20,17,-64,-3,-9,-21,-84]); //G
    let G=Polynomial::new(vec![-41,-34,-33,25,-41,31,-18,-32]); //G
    let h=Polynomial::new(vec![-4839,-6036,-4459,-2665,-186,-4303,3388,-3568]); //h
    let ntrukeys=falcon::NtruKeys::generate_lattice(f, g, F, G, h, phi);
    let message="Fuck you".to_string();
    let signature=ntrukeys.sign(message,beta);
    println!("{:?}",signature);
    
    let newmessage="Fuck you".to_string();
    let verification=ntrukeys.verify(newmessage,&signature);
    println!("{}",verification);
    */


    /*
    //test_matrix_multiplication_example();
    test_solve_linear_by_gaussian_elimination();
    */
    //test_calculate_matrix_inverse();
    //test_nearest_integer_mod_q();
    //test_solve_closest_vector_by_rounding_off();
    
    let beta = 1;
    let start_time = Instant::now();
    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1]); // φ = x^8 + 1
    let ntrukeys=falcon::NtruKeys::NTRUGen(&phi);
    let message="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
    let signature=ntrukeys.sign(message,beta);
    let time_taken = start_time.elapsed().as_millis();
    //let newmessage="aaaaaaaaaaaaaaa".to_string();
    //let verification=ntrukeys.verify(newmessage,&signature);
    //println!("{}",verification);

    println!("{}", time_taken);

}
*/

/*
fn main() {
    let beta = 1;
    let phi_sizes = vec![8, 16, 32, 64]; // φ sizes to test
    let message_sizes = vec![5,6]; // Message lengths to test

    // Prepare output file
    let file = File::create("experiment_results.csv").expect("Could not create file");
    let mut writer = BufWriter::new(file);

    // Write CSV header
    writeln!(writer, "phi_size,message_size,time_taken_ms,signature_size").expect("Failed to write to file");

    for phi_size in phi_sizes {
        // Create φ polynomial of specified size (e.g., x^(phi_size-1) + 1)
        let mut phi_coeffs = vec![0; phi_size+1];
        phi_coeffs[0] = 1;
        phi_coeffs[phi_size - 1] = 1;
        let phi = Polynomial::new(phi_coeffs);
        // Generate keys
        let ntrukeys = falcon::NtruKeys::NTRUGen(&phi);

        for message_size in &message_sizes {
            // Generate a message of specified size
            let message = "a".repeat(*message_size);

            // Measure time to sign
            let start_time = Instant::now();
            let signature = ntrukeys.sign(message.clone(), beta);
            let time_taken = start_time.elapsed().as_millis();

            // Get signature size
            let signature_size = signature.len();

            // Write results to file
            writeln!(writer, "{},{},{},{}", phi_size, message_size, time_taken, signature_size)
                .expect("Failed to write to file");
        }
    }

    println!("Experiment completed. Results saved to 'experiment_results.csv'.");
}
    */