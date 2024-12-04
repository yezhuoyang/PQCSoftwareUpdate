
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
    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1]); // φ = x^8 + 1
    let ntrukeys=falcon::NtruKeys::NTRUGen(&phi);

    

}
