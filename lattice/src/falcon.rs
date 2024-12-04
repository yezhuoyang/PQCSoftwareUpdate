use crate::poly::*;
use crate::config::*;
use std::fmt;
use std::io::{self, Read};
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use std::fs::File;
use rand::Rng;
use std::default::Default;
use rand_distr::{Distribution, Normal};
use ndarray::arr2;
use ndarray::Array2;


use crate::linearAlg::*;


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
    Q: i64,
    A: Vec<Vec<i64>>,
    B: Vec<Vec<i64>>
}



//Calculate the gram schmidt norm of the matrix B
//We have to ensure that |B|<=1.17q**0.5
fn GramSchmidtnorm(B:& Vec<Vec<i64>>)->f32{
    0.0
}

fn LDL(B:& Vec<Vec<i64>>)->Vec<Vec<i64>>{
    vec![]
}

fn ffLDL(B:& Vec<Vec<i64>>)->Vec<Vec<i64>>{
    vec![]
}


pub struct FalconTree{

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
            Q:q,
            A: vec![],
            B: vec![]
        }
    }
}

impl NtruKeys {
    /// Computes the public key h = g * f^{-1} mod φ mod q
    pub fn generate(f: Polynomial, g: Polynomial, phi: Polynomial) -> Self {
        let f_inv_mod_q = f.clone(); // Placeholder: Implement modular inversion
        let h = (g.clone() * f_inv_mod_q.clone()).mod_phi(&phi); // Clone `g` and `f_inv_mod_q`
        NtruKeys {
            f:f,
            g:g,
            f_inv_mod_q:f_inv_mod_q,
            h:h,
            ..Default::default() // Fill in the remaining members with default values
        }
    }


    pub fn generate_lattice(f: Polynomial, g: Polynomial, F: Polynomial, G: Polynomial, h:Polynomial,phi: Polynomial) -> Self{
        NtruKeys {
            f:f,
            F:F,
            g:g,
            G:G,
            h:h,
            phi:phi,
            ..Default::default() // Fill in the remaining members with default values
        }

    }


    //Generate the NTRU keys
    pub fn NTRUGen(phi: &Polynomial) -> Self {
        let sigma=100.0;
        let dimension=phi.degree()-1;
        println!("The phi is {}",phi);
        //Sampling two polynomials f and g from the distribution D_{\sigma}
        let f=generate_gaussian_polynomial(dimension,0.0,sigma);
        println!("The f is {}",f);
        let g=generate_gaussian_polynomial(dimension,0.0,sigma);
        println!("The g is {}",g);
        //Use extended Euclidean algorithm to get F and G
        let (G,minusF,gcd)=extended_gcd_poly(&f,&g);

        let F=minusF.multiply_by_scalar(-1).mod_phi(&phi);

        let Q=gcd.leading_coefficient();


        //Calculate the public key h, h=g*f^{-1} mod phi mod 1
        let finverse=f.inverse(&phi,&q);
        let h=(g.clone()*finverse.clone()).mod_phi(&phi);

        println!("The q is {}",q);

        println!("The F is {}",F);
        println!("The G is {}",G);
        println!("The q is {}",gcd);
        println!("h is {}",h);
        NtruKeys {
            f:f,
            F:F,
            g:g,
            G:G,
            Q:Q,
            phi:phi.clone(),
            ..Default::default() // Fill in the remaining members with default values
        }
    }


    // Solve the NTRU equation to get F and G, and get the public key h, secret key
    /// Solve the NTRU equation using the recursive approach described in Algorithm 6.
    pub fn NTRUSolve(f: &Polynomial, g: &Polynomial, phi: &Polynomial, Q: &i64) -> () {
        // Base case: if the degree of phi is 1 (n=1)
        // Use brute force to solve the equation
        // fG-gF=q mod phi
        // Two large matrices f and g, and two large matrices F and G are generated
        

    }

    // Add signature to the message using the secret key, bounded by beta
    // The idea is find the shortest vector in the lattice space spanned by A, with the help of the dual 
    // lattice B.
    pub fn sign(&self, message: String, beta: i64) -> Vec<i64>{
        let messagepoly=self.HashtoPoint(&message) as Polynomial;
        let messagevec=messagepoly.to_vec();
        println!("Hashing of message {:?}",messagevec);
        let Amatrix=calculate_public_key(&self.h,&self.phi);
        let A = array2_to_vecvec(Amatrix);
        let Bmatrix=calculate_secret_key(&self.f,&self.g,&self.G,&self.F,&self.phi);
        let B = array2_to_vecvec(Bmatrix);
        //Find one solution to the equation: A*s=messagevec
        let s=solve_linear_by_gaussian_elimination(&A,&messagevec,&q);
        let c0=solve_closest_vector_by_rounding_off(&B,&s,&q);
        //println!("The c0 is {:?}",c0);
        let result=vector_delete(&s,&c0,&self.Q);
        println!("The signature is {:?}",result);
        result
    }


    // Verify the signature using the public key
    pub fn verify(&self, message: String, signature: &Vec<i64>) -> bool{
        //First verify the correctness of the signature
        let messagepoly=self.HashtoPoint(&message) as Polynomial;
        let messagevec=messagepoly.to_vec();
        println!("Verifying message vec {:?}",messagevec);

        let Amatrix=calculate_public_key(&self.h,&self.phi);
        let A = array2_to_vecvec(Amatrix);
        
        let result=matrix_vector_multiplication(&A,signature,&q);
        if result==messagevec{
            return true;
        }
        false
    }


    // Convert the hash of the message to a polynomial
    pub fn HashtoPoint(&self, message: &String) -> Polynomial{
        let k = (1 << 16) / q; // 2^16 / q, rounded down (integer division)

        // Create a SHAKE-256 hasher
        let mut hasher = Shake256::default();
    
        // Feed the string content into the hasher
        hasher.update(message.as_bytes());
    
        // Obtain a reader for extracting pseudo-random bytes
        let mut reader = hasher.finalize_xof(); // Converts hasher to a SHAKE-256 XOF reader
    
        // Create a vector to store the coefficients with dimension n of the polynomial  
        let mut coefficients: Vec<i64> = vec![0; ndim];
        
        let mut i = 0;
        while i < ndim {
            // Extract 16 bits from the reader
            let mut buffer = [0u8; 2];
            reader.read_exact(&mut buffer).expect("Failed to read bytes");
    
            // Combine two bytes into a 16-bit integer
            let t = u16::from_le_bytes(buffer) as i64;
    
            // Filter values based on the condition
            if t < k * q {
                coefficients[i] = t;
                i += 1;
            }
        }
    
        // Create a polynomial with the extracted coefficients
        let p = Polynomial::new(coefficients);
        println!("{}", p);
        p
    }


    //Fast Fourier Sampling
    pub fn ffSampling(&self, beta: i64) -> Polynomial{
        self.f.clone()
    }


    pub fn BaseSampler(&self) -> i64{
        0
    }

    //An approximation of 2^{63}*ccs*e^{-x}
    pub fn ApproxExp(x:f64,ccs:f64) -> f64{
        0.0
    }

    //Return a single bit, which equals 1 with probability ccs*e^{-x}
    pub fn BerExp(x:f64,ccs:f64) -> i64{
        0
    }


    pub fn SamplerZ(&self, mu: f64,sigma: f64) -> i64{
        let r=mu - mu.floor();
        let ccs=sigmamin/sigma;
        0
    }

    pub fn splitfft(&self, beta: i64) -> Polynomial{
        self.f.clone()
    }


    pub fn mergefft(&self, beta: i64) -> Polynomial{
        self.f.clone()
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
pub fn verify_lattice_orthorgonal(Amat:ndarray::Array2<i64>, Bmat:ndarray::Array2<i64>) -> bool{
    let Amat_shape = Amat.shape();
    let Bmat_shape = Bmat.shape();
    if Bmat_shape[1] != Amat_shape[1] {
        panic!(
            "Matrix multiplication dimensions mismatch: Bmat is {:?}, Amat is {:?}",
            Bmat_shape, Amat_shape
        );
    }
    let Cmat=Bmat.dot(&Amat.t());



    let C=Cmat.iter().map(|&x| x%q).collect::<Vec<i64>>();
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
pub fn calculate_secret_key(f: &Polynomial, g: &Polynomial, G:&Polynomial, F:&Polynomial, phi:&Polynomial) -> ndarray::Array2<i64> {
   let fmat=f.to_ndarray(&phi);
   let gmat=g.to_ndarray(&phi);
   let Gmat=G.to_ndarray(&phi);
   let Fmat=F.to_ndarray(&phi);
   // Combine four matrices to get [[-gmat,-fmat],[G,-F]]
   let mut B=Array2::<i64>::zeros((2*ndim,2*ndim));
    for i in 0..ndim{
         for j in 0..ndim{
              B[[i,j]]=gmat[[i,j]];
              B[[i,j+ndim]]=-fmat[[i,j]];
              B[[i+ndim,j]]=Gmat[[i,j]];
              B[[i+ndim,j+ndim]]=-Fmat[[i,j]];
         }
    }
    B
}

pub fn calculate_public_key(h: &Polynomial, phi:&Polynomial) -> ndarray::Array2<i64> {
    let hmat=h.to_ndarray(phi);
    // Construct new matrix [1|hmat], 1 is the identity of n*n matrix
    let mut A=Array2::<i64>::zeros((ndim,2*ndim));
    for i in 0..ndim{
        for j in 0..ndim{
            if i==j{
                A[[i,j]]=1;
            }
            else{
                A[[i,j]]=0;
            }
            A[[i,j+ndim]]=hmat[[j,i]];
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
//Return a polynomial with coefficients sampled from a Gaussian distribution
pub fn generate_gaussian_polynomial(n: usize, mean: f64, std_dev: f64) -> Polynomial {
    // Create a normal distribution with the specified mean and standard deviation
    let normal = Normal::new(mean, std_dev).unwrap();

    // Initialize random number generator
    let mut rng = rand::thread_rng();

    // Generate coefficients
    let coefficients=(0..=n)
        .map(|_| normal.sample(&mut rng).round() as i64) // Sample and round to nearest integer
        .collect();
    Polynomial::new(coefficients)
}


//Test the validity of the f and g
//f should be invertible
pub fn test_validity_of_fg(f:&Polynomial, g:&Polynomial)->bool{

    true
}


#[cfg(test)]
mod tests {
    use super::*; // Import all items from the parent module


    #[test]
    fn test_key_generation_example1(){

        let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1]); // φ = x^8 + 1
        let f=Polynomial::new(vec![-55,11,-23,-23,47,16,13,61]); //f
        let g=Polynomial::new(vec![-25,-24,30,-3,36,-39,6]); //g
        let F=Polynomial::new(vec![58,20,17,-64,-3,-9,-21,-84]); //G
        let G=Polynomial::new(vec![-41,-34,-33,25,-41,31,-18,-32]); //G
        let h=Polynomial::new(vec![-4839,-6036,-4459,-2665,-186,-4303,3388,-3568]); //h
    
        let B=calculate_secret_key(&f,&g,&G,&F,&phi);
        let A=calculate_public_key(&h,&phi);
    
        let result=verify_lattice_orthorgonal(A,B);
        assert_eq!(result,true);
    }



}
