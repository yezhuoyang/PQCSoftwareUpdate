use crate::poly::Polynomial;
use std::io::{self, Read};
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use std::fs::File;
use rand::Rng;
use rand_distr::{Distribution, Normal};


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