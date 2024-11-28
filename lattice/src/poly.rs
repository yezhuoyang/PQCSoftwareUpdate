use std::fmt;
use std::ops::{Add, Mul, Sub, Rem};
use std::default::Default;
use std::f64::consts::PI;
use crate::config::*;
use ndarray::Array2;
use num::complex::Complex;



/// Represents a polynomial over Z_q[x]
#[derive(Debug, Clone)]
pub struct Polynomial {
    coefficients: Vec<i32>, // Coefficients of the polynomial
}

impl Polynomial {
    /// Creates a new polynomial with given coefficients and modulus
    /// The coefficients are reduced modulo q, i.e., coefficients[i] = coefficients[i] % q
    /// The coefficients are put in rising order of degrees. For example, [1, 2, 3] represents 1 + 2x + 3x^2
    pub fn new(coefficients: Vec<i32>) -> Self {
        let coefficients = coefficients.into_iter().map(|c| c.rem_euclid(q)).collect();
        Polynomial { coefficients}
    }


    /// Degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
           0
        }
        else{
            self.coefficients.len() - 1
        }
    }

    /// Shifts a polynomial by a given number of degrees (multiply by x^degree),
    pub fn shift(&self, degree: usize)-> Polynomial{
        // Create a shifted polynomial by appending `degree` zeros
        let mut new_coeffs = vec![0; degree];
        new_coeffs.extend(&self.coefficients);

        // Create a new Polynomial object
        let shifted_poly = Polynomial::new(new_coeffs);
        shifted_poly
    }

    /// Return a new polynomial, which is self-other
    pub fn delete(&self, other: &Polynomial) -> Polynomial{
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
                new_coeffs[index]=(self_coeffs[index]-other_coeffs[index]).rem_euclid(q);
        }    
        let mut newpoly=Polynomial::new(new_coeffs);
        newpoly.clear_zeros(); 
        newpoly
    }


    //Modulus of the polynomial
    pub fn mod_phi(&self, phi: &Polynomial) -> Polynomial {
        let phideg=phi.degree();
        let selfdegree: usize=self.degree();
        if selfdegree<phideg{
            return self.clone();
        }
        let gap=selfdegree-phideg;
        let selfcoeffts=self.coefficients.clone();
        let selflast = *selfcoeffts.last().unwrap(); // Extract the value and dereference it
        let newpoly=self.delete(&phi.shift(gap).multiple(selflast));
        newpoly.mod_phi(phi)
    }
    
    
    
    pub fn to_matrix(&self, phi: &Polynomial) -> Vec<Vec<i32>> {
        let n = phi.degree(); // Degree of φ
        let mut matrix = vec![vec![0; n]; n]; // Initialize an n x n matrix
    
        for i in 0..n {
            // Pass both the shift degree and the modulus polynomial
            let shifted_poly = self.shift(i);
            let reduced_poly = shifted_poly.mod_phi(phi);
            /*
            println!(
                "x^{} * f = {:?} mod φ = {:?}",
                i, shifted_poly.coefficients, reduced_poly.coefficients
            );

            println!(
                "x^{} * f = {} mod φ = {}",
                i, &shifted_poly, &reduced_poly
            );
            */
            // Fill the row with the reduced coefficients
            for (j, &coeff) in reduced_poly.coefficients.iter().enumerate() {
                matrix[j][i] = coeff;
            }
        }
        matrix
    }


    
    pub fn to_ndarray(&self, phi: &Polynomial) -> ndarray::Array2<i32> {
        let n = phi.degree(); // Degree of φ
        let mut matrix = ndarray::Array2::<i32>::zeros((n, n)); // Initialize an n x n matrix
    
        for i in 0..n {
            // Pass both the shift degree and the modulus polynomial
            let shifted_poly = self.shift(i);
            let reduced_poly = shifted_poly.mod_phi(phi);
            /*
            println!(
                "x^{} * f = {:?} mod φ = {:?}",
                i, shifted_poly.coefficients, reduced_poly.coefficients
            );

            println!(
                "x^{} * f = {} mod φ = {}",
                i, &shifted_poly, &reduced_poly
            );
            */
            // Fill the row with the reduced coefficients
            for (j, &coeff) in reduced_poly.coefficients.iter().enumerate() {
                matrix[[j, i]] = coeff;
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

    pub fn multiple(&self, value: i32) -> Polynomial {
        let new_coeffs: Vec<i32> = self
            .coefficients
            .iter()
            .map(|&coeff| coeff * value)
            .collect();
    
        Polynomial::new(new_coeffs)
    }

    pub fn equal(&self, other: &Polynomial, phi: &Polynomial) -> bool {
        let newpoly=self.clone()-other.clone();
        if newpoly.mod_phi(phi).degree()==0{
            true
        }
        else{
            false
        }
    }

    //Calculate the inverse of the polynomial modulo phi
    pub fn inverse(&self,phi: &Polynomial)-> Polynomial{
        self.clone()
    }

    //Calculate the value of the polynomial at a given point x
    pub fn calculate_value_int(&self, x: i32) -> i32 {
        let mut result = 0;
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            result += coeff * x.pow(i as u32);
        }
        result.rem_euclid(q)
    }

    //Calculate the value of the polynomial of a floating point value x
    pub fn calculate_value_float(&self, x:f64)-> f64{
        let mut result=0.0;
        for (i,&coeff) in self.coefficients.iter().enumerate(){
            result+=coeff as f64*x.powi(i as i32);
        }
        result
    }

    //Calculate the value of the polynomial of complex value a+bi
    pub fn calculate_value_complex(&self, x: Complex<f64>) -> Complex<f64> {
        let mut result = Complex::new(0.0, 0.0);
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            result += Complex::new(coeff as f64, 0.0) * x.powf(i as f64);
        }
        result
    }


}


impl fmt::Display for Polynomial{
    fn fmt(&self, f: &mut fmt::Formatter<'_>)-> fmt::Result{
        write!(f,"Polynomial (mod {}): ", q)?;
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





//Operator overloading for subtraction, p1*p2
impl Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: Polynomial) -> Polynomial {
        let mut result = vec![0; self.coefficients.len() + rhs.coefficients.len() - 1];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in rhs.coefficients.iter().enumerate() {
                result[i + j] = (result[i + j] + a * b).rem_euclid(q);
            }
        }
        Polynomial::new(result)
    }
}



//Operator overloading for subtraction, p1+p2
impl Add for Polynomial{
    type Output = Polynomial;
    fn add(self, rhs: Polynomial) -> Polynomial{
         let selfdegree=self.degree();
         let rhsdegree=rhs.degree();
         let newdegree=std::cmp::max(selfdegree,rhsdegree);
         let mut newcoeffs=vec![0;newdegree+1];
        for i in 0..newdegree+1{
            let selfval=self.coefficients.get(i).unwrap_or(&0);
            let rhsval=rhs.coefficients.get(i).unwrap_or(&0);
            newcoeffs[i]=(*selfval+*rhsval).rem_euclid(q);
        }
        Polynomial::new(newcoeffs)
    }
}




//Operator overloading for subtraction, p1-p2
impl Sub for Polynomial{
    type Output = Polynomial;
    fn sub(self, rhs: Polynomial) -> Polynomial{
         let selfdegree=self.degree();
         let rhsdegree=rhs.degree();
         let newdegree=std::cmp::max(selfdegree,rhsdegree);
         let mut newcoeffs=vec![0;newdegree+1];
        for i in 0..newdegree+1{
            let selfval=self.coefficients.get(i).unwrap_or(&0);
            let rhsval=rhs.coefficients.get(i).unwrap_or(&0);
            newcoeffs[i]=(*selfval-*rhsval).rem_euclid(q);
        }
        Polynomial::new(newcoeffs)
    }
}



impl Default for Polynomial {
    fn default() -> Self {
        Polynomial {
            coefficients: vec![]
        }
    }
}


//Compress the polynomial to a string
pub fn compress(poly: &Polynomial) -> String{
    let mut compressed=String::new();
    compressed
}

//Decompress the polynomial from a string
pub fn decompress(compressed: &String)-> Polynomial{
    let mut decompressed=Polynomial::default();
    decompressed
}



//Calculate all the unit roots of a given order n
//The equation is x^n=1
pub fn calculate_unit_roots(n: &i32) -> Vec<(f64, f64)> {
    let n_as_f64 = *n as f64; // Convert the value of n to f64
    (0..*n) // Use n directly here as an integer
        .map(|k| {
            let angle = 2.0 * PI * k as f64 / n_as_f64; // Calculate the angle
            (angle.cos(), angle.sin()) // Return (cos(angle), sin(angle)) as a tuple
        })
        .collect() // Collect the results into a vector
}



//The FFT representation of a polynomial modulo phi, which is a vector of complex numbers, stored as a vector of f64
pub fn FFT(poly: &Polynomial, phi: &Polynomial)-> Vec<f64>{
    let phideg=phi.degree();
    //Store the complex numbers as a vector of f64
    let mut fft=vec![0.0;2*phideg];
    let unit_roots=calculate_unit_roots(&(phideg as i32));
    for i in 0..phideg{
        let root=unit_roots[i];
        let tmpcomplex=Complex::new(root.0,root.1);
        fft[i]=tmpcomplex.re;
        fft[i+phideg]=tmpcomplex.im;
    }
    fft
}


//Create a Vandermonde matrix from a given input vector
fn vandermonde_matrix(input: &[f64], n: usize) -> Array2<Complex<f64>> {
    let mut matrix = Array2::<Complex<f64>>::zeros((n, n));
    for i in 0..n{
        let root=Complex::new(input[i],input[i+n]);
        for j in 0..n{
            matrix[[i,j]]=root.powf(j as f64);
        }
    }
    matrix
}


/*
pub fn inverseFFT(phi: &Polynomial, fft: &Vec<f64>)->Polynomial{

    Polynomial::new(poly)
}
*/


//The NTT representation of a polynomial modulo phi
pub fn NTT(poly: &Polynomial, phi: &Polynomial)-> Vec<i32>{
    let n=phi.degree();
    let mut ntt=vec![0;2*n];
    ntt
}



#[cfg(test)]
mod tests {
    use super::*; // Import all items from the parent module

    #[test]
    fn test_polynomial_to_matrix() {
        // Define the modulus
        let phi = Polynomial::new(vec![1, 0, 0, 0, 1]); // φ = x^4 + 1

        // Define the polynomial f = x^2 + 1
        let f = Polynomial::new(vec![1, 0, 1]); // Coefficients: [0, 0, 1, 0, 1]

        // Generate the matrix representation of f
        let matrix = f.to_matrix(&phi);

        // Expected result
        let expected_matrix = vec![
            vec![1, 0, 4, 0],
            vec![0, 1, 0, 4],
            vec![1, 0, 1, 0],
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
