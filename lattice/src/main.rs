use std::ops::{Add, Mul, Rem};
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use std::fs::File;
use std::io::{self, Read};

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

    /// Polynomial modulo φ(x)
    fn mod_phi(&self, phi: &Polynomial) -> Polynomial {
        let mut result = self.clone();
        while result.degree() >= phi.degree() {
            let factor = result.coefficients[result.degree()] / phi.coefficients[phi.degree()];
            for (i, &coeff) in phi.coefficients.iter().enumerate() {
                let index = result.degree() - phi.degree() + i;
                result.coefficients[index] =
                    (result.coefficients[index] - factor * coeff).rem_euclid(self.q);
            }
            while let Some(&last) = result.coefficients.last() {
                if last == 0 {
                    result.coefficients.pop();
                } else {
                    break;
                }
            }
        }
        result
    }

    /// Generate the matrix representation of the polynomial
    fn to_matrix(&self, phi: &Polynomial) -> Vec<Vec<i32>> {
        let n = phi.degree();
        let mut matrix = vec![vec![0; n]; n];

        for i in 0..n {
            let shifted_poly = Polynomial::new(
                self.coefficients.iter().cycle().skip(i).take(n).cloned().collect(),
                self.q,
            );
            let row = shifted_poly.mod_phi(phi).coefficients;
            for (j, &coeff) in row.iter().enumerate() {
                matrix[i][j] = coeff;
            }
        }

        matrix
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