
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



fn read_string_from_file(file_path: &str) -> Result<String, io::Error> {
    fs::read_to_string(file_path)
}







fn main() {
    //test_poly_deletion();
    //test_poly_equal();
    //test_poly_sum();
    //test_poly_multiplication();
    //extended_gcd_poly_test();
    //extended_gcd_poly_example();
    //test_leading_coeff()
    //test_determinant_gaussian_elimination();
    //test_poly_inverse();
    //test_inverse_mod();
    //extended_gcd_poly_example();
    test_solve_linear_by_gaussian_elimination();
    //test_solve_linear_example();
}
