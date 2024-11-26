use crate::poly::Polynomial;
use std::fmt;
use std::io::{self, Read};
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use std::fs::File;
use rand::Rng;
use std::default::Default;
use rand_distr::{Distribution, Normal};
use ndarray::arr2;
use ndarray::Array2;


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
pub struct NtruKeys {
    f: Polynomial,
    F: Polynomial,
    g: Polynomial,
    G: Polynomial,
    phi: Polynomial,
    f_inv_mod_q: Polynomial,
    h: Polynomial,
    A: Vec<Vec<i32>>,
    B: Vec<Vec<i32>>,
    q: i32
}

impl Default for NtruKeys {
    fn default() -> Self {
        NtruKeys {
            f: Polynomial::default(),
            F: Polynomial::default(),
            g: Polynomial::default(),
            G: Polynomial::default(),
            phi: Polynomial::default(),
            f_inv_mod_q: Polynomial::default(),
            h: Polynomial::default(),
            A: vec![],
            B: vec![],
            q: 12289
        }
    }
}

impl NtruKeys {
    /// Computes the public key h = g * f^{-1} mod φ mod q
    pub fn generate(f: Polynomial, g: Polynomial, phi: Polynomial, _q: i32) -> Self {
        let f_inv_mod_q = f.clone(); // Placeholder: Implement modular inversion
        let h = (g.clone() * f_inv_mod_q.clone()).mod_phi(&phi); // Clone `g` and `f_inv_mod_q`

        NtruKeys {
            f:f,
            g:g,
            f_inv_mod_q:f_inv_mod_q,
            h:h,
            q:_q,
            ..Default::default() // Fill in the remaining members with default values
        }
    }


    pub fn generate_lattice(f: Polynomial, g: Polynomial, F: Polynomial, G: Polynomial, h:Polynomial,phi: Polynomial,_q: i32) -> Self{

        NtruKeys {
            f:f,
            F:F,
            g:g,
            G:G,
            h:h,
            phi:phi,
            q:_q,
            ..Default::default() // Fill in the remaining members with default values
        }

    }




    // Solve the NTRU equation to get F and G, and get the public key h, secret key
    pub fn solveNTRU(self, f: Polynomial, g: Polynomial, phi: Polynomial, _q: i32) -> (){

    }

    // Add signature to the message using the secret key
    // The idea is find the shortest vector in the lattice space spanned by A, with the help of the dual 
    // lattice B.
    pub fn sign(self, message: String) -> String{
        "22".to_string()
    }


    // Verify the signature using the public key
    pub fn verify(self, message: String, signature: String) -> bool{
        true
    }


}


impl fmt::Display for NtruKeys{
    fn fmt(&self, f: &mut fmt::Formatter<'_>)-> fmt::Result{
        write!(f,"NTRUKeys:\n")?;
        write!(f,"f: {}\n", self.f)?;
        write!(f,"F: {}\n", self.F)?;
        write!(f,"g: {}\n", self.g)?;
        write!(f,"G: {}\n", self.G)?;
        write!(f,"phi: {}\n", self.phi)?;
        write!(f,"f_inv_mod_q: {}\n", self.f_inv_mod_q)?;
        write!(f,"h: {}\n", self.h)?;
        write!(f,"A: {:?}\n", self.A)?;
        write!(f,"B: {:?}\n", self.B)?;    
        Ok(())     
    }
}




/// Verifies that the lattice A is orthogonal to the lattice B.
pub fn verify_lattice_orthorgonal(Amat:ndarray::Array2<i32>, Bmat:ndarray::Array2<i32>, q:i32) -> bool{
    let Amat_shape = Amat.shape();
    let Bmat_shape = Bmat.shape();
    if Bmat_shape[1] != Amat_shape[1] {
        panic!(
            "Matrix multiplication dimensions mismatch: Bmat is {:?}, Amat is {:?}",
            Bmat_shape, Amat_shape
        );
    }
    let Cmat=Bmat.dot(&Amat.t());



    let C=Cmat.iter().map(|&x| x%q).collect::<Vec<i32>>();
    println!("{:?}",C);

    for i in 0..C.len(){
        //Every element should be 0
        if C[i]!=0{
            return false;
        }
    }
    true
}


//Calculate the secret key, which is an ndarray matrix
pub fn calculate_secret_key(f: &Polynomial, g: &Polynomial, G:&Polynomial, F:&Polynomial, phi:&Polynomial,q: i32) -> ndarray::Array2<i32> {
   let fmat=f.to_ndarray(phi);
   let gmat=g.to_ndarray(phi);
   let Gmat=G.to_ndarray(phi);
   let Fmat=F.to_ndarray(phi);
   let n=fmat.shape()[0];
   // Combine four matrices to get [[-gmat,-fmat],[G,-F]]
   let mut B=Array2::<i32>::zeros((2*n,2*n));
    for i in 0..n{
         for j in 0..n{
              B[[i,j]]=gmat[[i,j]];
              B[[i,j+n]]=-fmat[[i,j]];
              B[[i+n,j]]=Gmat[[i,j]];
              B[[i+n,j+n]]=-Fmat[[i,j]];
         }
    }
    B
}

pub fn calculate_public_key(h: &Polynomial, Phi:&Polynomial, q: i32) -> ndarray::Array2<i32> {
    let hmat=h.to_ndarray(Phi);
    let n=hmat.shape()[0];
    // Construct new matrix [1|hmat], 1 is the identity of n*n matrix
    let mut A=Array2::<i32>::zeros((n,2*n));
    for i in 0..n{
        for j in 0..n{
            if i==j{
                A[[i,j]]=1;
            }
            else{
                A[[i,j]]=0;
            }
            A[[i,j+n]]=hmat[[j,i]];
        }
    }
    A
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
pub fn generate_gaussian_polynomial(n: usize, mean: f64, std_dev: f64) -> Vec<i32> {
    // Create a normal distribution with the specified mean and standard deviation
    let normal = Normal::new(mean, std_dev).unwrap();

    // Initialize random number generator
    let mut rng = rand::thread_rng();

    // Generate coefficients
    (0..=n)
        .map(|_| normal.sample(&mut rng).round() as i32) // Sample and round to nearest integer
        .collect()
}