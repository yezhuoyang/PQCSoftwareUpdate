use std::ops::{Add, Mul, Rem};
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use std::fs::File;
use std::io::{self, Read};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fmt;

/// Reads a file in binary mode and returns its SHAKE-256 hash.
fn hash_file_with_shake256(file_path: &str, hash_output_size: usize) -> io::Result<String> {
    // Open the file in binary mode
    let mut file = File::open(file_path)?;

    // Create a buffer to read the file contents
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Create a SHAKE-256 hasher
    let mut hasher = Shake256::default();

    // Feed the file content into the hasher
    hasher.update(&buffer);

    // Finalize the hasher and extract the hash output
    let mut reader = hasher.finalize_xof();
    let mut hash = vec![0u8; hash_output_size];
    reader.read_exact(&mut hash)?;

    // Convert the hash output to a hexadecimal string
    Ok(hash.iter().map(|b| format!("{:02x}", b)).collect())
}


/// Represents a polynomial over Z_q[x]
#[derive(Debug, Clone)]
struct Polynomial {
    coefficients: Vec<i32>, // Coefficients of the polynomial
    q: i32,                // Modulus
}

impl Polynomial {
    /// Creates a new polynomial with given coefficients and modulus
    fn new(coefficients: Vec<i32>, q: i32) -> Self {
        let coefficients = coefficients.into_iter().map(|c| c.rem_euclid(q)).collect();
        Polynomial { coefficients, q }
    }


    /// Degree of the polynomial
    fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Shifts a polynomial by a given number of degrees (multiply by x^degree),
    fn shift(&self, degree: usize)-> Polynomial{
        // Create a shifted polynomial by appending `degree` zeros
        let mut new_coeffs = vec![0; degree];
        new_coeffs.extend(&self.coefficients);

        // Create a new Polynomial object
        let shifted_poly = Polynomial::new(new_coeffs, self.q);
        shifted_poly
    }

    /// Return a new polynomial, which is self-other
    fn delete(&self, other: &Polynomial) -> Polynomial{
        let mut self_coeffs=self.coefficients.clone();
        let mut other_coeffs=other.coefficients.clone();
        //Determine which polynomial is longer, append zeros to the shorter one for deletion
        let gap=self_coeffs.len()  as isize  -other_coeffs.len() as isize;
        let mut zero_coeffs = vec![0; gap.abs() as usize];
        if gap>=0{
             other_coeffs.append(&mut zero_coeffs);
        }
        else{
             self_coeffs.append(&mut zero_coeffs);
        }
        let mut new_coeffs = vec![0; self_coeffs.len()];
        let index=0;
        for index in 0..self_coeffs.len(){
                new_coeffs[index]=(self_coeffs[index]-other_coeffs[index]).rem_euclid(self.q);
        }    
        let mut newpoly=Polynomial::new(new_coeffs, self.q);
        newpoly.clear_zeros(); 
        newpoly
    }


    //Modulus of the polynomial
    fn mod_phi(&self, phi: &Polynomial) -> Polynomial {
        let phideg=phi.degree();
        
    }
    
    
    
    fn to_matrix(&self, phi: &Polynomial) -> Vec<Vec<i32>> {
        let n = phi.degree(); // Degree of φ
        let mut matrix = vec![vec![0; n]; n]; // Initialize an n x n matrix
    
        for i in 0..n {
            // Pass both the shift degree and the modulus polynomial
            let shifted_poly = self.shift(i);
            let reduced_poly = shifted_poly.mod_phi(phi);
    
            println!(
                "x^{} * f = {:?} mod φ = {:?}",
                i, shifted_poly.coefficients, reduced_poly.coefficients
            );
    
            // Fill the row with the reduced coefficients
            for (j, &coeff) in reduced_poly.coefficients.iter().enumerate() {
                matrix[i][j] = coeff;
            }
        }
        matrix
    }
    

    //Clear the zeros in the coefficient from the highest orders
    //Forexample: 0x^3+x^2-> x^2 since 0x^3 is unnecessary
    fn clear_zeros(&mut self){
        while let Some(&last)=self.coefficients.last(){
            if last!=0{
                break;
            }
            self.coefficients.pop();
        }
    }

    fn multiple(&self, value: i32) -> Polynomial {
        let new_coeffs: Vec<i32> = self
            .coefficients
            .iter()
            .map(|&coeff| coeff * value)
            .collect();
    
        Polynomial::new(new_coeffs, self.q)
    }
    

}


impl fmt::Display for Polynomial{
    fn fmt(&self, f: &mut fmt::Formatter<'_>)-> fmt::Result{
        write!(f,"Polynomial (mod {}): ", self.q)?;
        let mut first=true;
        for(i,&coef) in self.coefficients.iter().enumerate().rev(){
            if coef !=0{
                if !first {
                    write!(f, " + ")?;
                }
                if i == 0 {
                    write!(f, "{}", coef)?;
                } else if i == 1 {
                    write!(f, "{}x", coef)?;
                } else {
                    write!(f, "{}x^{}", coef, i)?;
                }
                first = false;               
            }

        }
        Ok(())       
    }
}





/// Implement multiplication for Polynomial
impl Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: Polynomial) -> Polynomial {
        let mut result = vec![0; self.coefficients.len() + rhs.coefficients.len() - 1];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in rhs.coefficients.iter().enumerate() {
                result[i + j] = (result[i + j] + a * b).rem_euclid(self.q);
            }
        }
        Polynomial::new(result, self.q)
    }
}


fn test_polynomial_to_matrix() {
    // Define the modulus
    let phi = Polynomial::new(vec![1, 0, 0, 0, 1], 5); // φ = x^4 + 1

    // Define the polynomial f = x^2 + 1
    let f = Polynomial::new(vec![0, 0, 1, 0, 1], 5); // Coefficients: [0, 0, 1, 0, 1]

    // Generate the matrix representation of f
    let matrix = f.to_matrix(&phi);

    // Expected result
    let expected_matrix = vec![
        vec![1, 0, 4, 0],
        vec![0, 1, 0, 4],
        vec![1, 0, 2, 0],
        vec![0, 1, 0, 1],
    ];

    // Print the result
    println!("Computed matrix:");
    for row in &matrix {
        println!("{:?}", row);
    }

    // Verify correctness
    assert_eq!(matrix, expected_matrix, "Matrix representation is incorrect");
}


/// Represents the NTRU lattice keys
struct NtruKeys {
    f: Polynomial,
    g: Polynomial,
    f_inv_mod_q: Polynomial,
    h: Polynomial,
}

impl NtruKeys {
    /// Computes the public key h = g * f^{-1} mod φ mod q
    fn generate(f: Polynomial, g: Polynomial, phi: Polynomial, _q: i32) -> Self {
        let f_inv_mod_q = f.clone(); // Placeholder: Implement modular inversion
        let h = (g.clone() * f_inv_mod_q.clone()).mod_phi(&phi); // Clone `g` and `f_inv_mod_q`

        NtruKeys {
            f,
            g,
            f_inv_mod_q,
            h,
        }
    }
}



/// Generates a polynomial with coefficients sampled from a Gaussian distribution.
/// 
/// # Arguments:
/// - `n`: The degree of the polynomial.
/// - `mean`: The mean of the Gaussian distribution (usually 0).
/// - `std_dev`: The standard deviation of the Gaussian distribution.
/// 
/// # Returns:
/// - A vector of sampled coefficients.
fn generate_gaussian_polynomial(n: usize, mean: f64, std_dev: f64) -> Vec<i32> {
    // Create a normal distribution with the specified mean and standard deviation
    let normal = Normal::new(mean, std_dev).unwrap();

    // Initialize random number generator
    let mut rng = rand::thread_rng();

    // Generate coefficients
    (0..=n)
        .map(|_| normal.sample(&mut rng).round() as i32) // Sample and round to nearest integer
        .collect()
}




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



#[cfg(test)]
mod tests {
    use super::*; // Import all items from the parent module

    #[test]
    fn test_polynomial_to_matrix() {
        // Define the modulus
        let phi = Polynomial::new(vec![1, 0, 0, 0, 1], 5); // φ = x^4 + 1

        // Define the polynomial f = x^2 + 1
        let f = Polynomial::new(vec![0, 0, 1, 0, 1], 5); // Coefficients: [0, 0, 1, 0, 1]

        // Generate the matrix representation of f
        let matrix = f.to_matrix(&phi);

        // Expected result
        let expected_matrix = vec![
            vec![1, 0, 4, 0],
            vec![0, 1, 0, 4],
            vec![1, 0, 2, 0],
            vec![0, 1, 0, 1],
        ];

        // Print the result (optional, for debugging purposes)
        println!("Computed matrix:");
        for row in &matrix {
            println!("{:?}", row);
        }

        // Verify correctness
        assert_eq!(matrix, expected_matrix, "Matrix representation is incorrect");
    }
}



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
    let poly1=Polynomial::new(vec![4,0,2,0,1],5);
    println!("{}",&poly1);
    let poly2=poly1.multiple(2);
    println!("{}",&poly2);   

}