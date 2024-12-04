use std::fmt;
use std::ops::{Add, Mul, Sub, Rem};
use std::default::Default;
use std::f64::consts::PI;
use crate::config::*;


use crate::linearAlg::*;
use crate::number::*;

use ndarray::Array2;
use num::complex::Complex;
//use ndarray_linalg::solve::Determinant;
use ndarray::Axis;
use rand::Rng;




/// Represents a polynomial over Z_q[x]
#[derive(Debug, Clone)]
pub struct Polynomial {
    coefficients: Vec<i64>, // Coefficients of the polynomial
}



impl Polynomial {
    /// Creates a new polynomial with given coefficients and modulus
    /// The coefficients are reduced modulo q, i.e., coefficients[i] = coefficients[i] % q
    /// The coefficients are put in rising order of degrees. For example, [1, 2, 3] represents 1 + 2x + 3x^2
    pub fn new(coefficients: Vec<i64>) -> Self {
        let coefficients = coefficients.into_iter().map(|c| c.rem_euclid(q)).collect();
        Polynomial { coefficients}
    }


    pub fn to_vec(&self) -> Vec<i64> {
        self.coefficients.clone()
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

    pub fn leading_coefficient(&self) -> i64 {
        let mut tmp=self.clone();
        tmp.clear_zeros();
        if tmp.coefficients.is_empty() {
            0
        }
        else{
            *tmp.coefficients.last().unwrap()
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
                new_coeffs[index]=(self_coeffs[index]-other_coeffs[index]);
        }    
        let mut newpoly=Polynomial::new(new_coeffs);
        newpoly.clear_zeros(); 
        newpoly
    }



    pub fn multiply_by_scalar(&self, scalar: i64) -> Polynomial {
        let new_coeffs: Vec<i64> = self
            .coefficients
            .iter()
            .map(|&coeff| coeff * scalar)
            .collect();
    
        Polynomial::new(new_coeffs)
    }


    //Modulus of the polynomial
    //Note that phi is monic, i.e., the leading coefficient is 1
    pub fn mod_phi(&self, phi: &Polynomial) -> Polynomial {
        let phideg=phi.degree();
        let selfdegree: usize=self.degree();
        if selfdegree<phideg{
            let mut result=self.clone();
            result.clear_zeros();
            return result;
        }
        let gap=selfdegree-phideg;
        let selfcoeffts=self.coefficients.clone();
        let selflast = *selfcoeffts.last().unwrap(); // Extract the value and dereference it
        let newpoly=self.delete(&phi.shift(gap).multiple(selflast));
        newpoly.mod_phi(phi)
    }
    
    
    
    pub fn to_matrix(&self, phi: &Polynomial) -> Vec<Vec<i64>> {
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


    
    pub fn to_ndarray(&self, phi: &Polynomial) -> ndarray::Array2<i64> {
        let n = phi.degree(); // Degree of φ
        let mut matrix = ndarray::Array2::<i64>::zeros((n, n)); // Initialize an n x n matrix
    
        for i in 0..n {
            // Pass both the shift degree and the modulus polynomial
            let shifted_poly = self.shift(i);
            let reduced_poly = shifted_poly.mod_phi(phi);
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

    pub fn multiple(&self, value: i64) -> Polynomial {
        let new_coeffs: Vec<i64> = self
            .coefficients
            .iter()
            .map(|&coeff| coeff * value)
            .collect();
    
        Polynomial::new(new_coeffs)
    }

    //Equality of two polynomials modulo phi
    pub fn equal(&self, other: &Polynomial, phi: &Polynomial) -> bool {
        let newpoly=self.clone()-other.clone();
        let modpoly=newpoly.mod_phi(phi);
        if newpoly.mod_phi(phi).degree()==0{
            true
        }
        else{
            false
        }
    }

    //Equality of two polynomials
    //Simply compare all coefficients
    pub fn equal_exact(&self, other: &Polynomial) -> bool{
        let selfdegree=self.degree();
        let otherdegree=other.degree();
        if selfdegree!=otherdegree{
            return false;
        }
        for i in 0..selfdegree{
            if self.coefficients[i]!=other.coefficients[i]{
                return false;
            }
        }
        true
    }

    //Calculate the inverse of the polynomial modulo phi
    //Here we use extended GCD algorithm to calculate the inverse
    //The inverse of f is g such that f*g=1 mod phi
    pub fn inverse(&self,phi: &Polynomial)-> Polynomial{
        let (a,_,gcd)=extended_gcd_poly(self,phi);
        if gcd.degree()!=0{
            panic!("The polynomial is not invertible");
        }
        let gcdvalue=gcd.leading_coefficient();
        if gcdvalue==0{
            panic!("Wrong phi!");
        }
        let gcdvalue_inverse=inverse_mod(gcdvalue,q);
        a.multiple(gcdvalue_inverse)
    }

    //Calculate the value of the polynomial at a given point x
    pub fn calculate_value_int(&self, x: i64) -> i64 {
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

    //Calculate the resultant of two polynomials
    //Here, phi is monic, i.e., the leading coefficient is 1, so the resultant can be 
    //calculated as the determinant of the Sylvester matrix
    pub fn resultant(&self, phi: &Polynomial) -> i64 {
        let syl=self.to_ndarray(phi);
        //Convert syl to a ndarray stored with f64
        let det=gaussian_elimination_determinant(syl);
        det.rem_euclid(q)
    }



    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }



    //Calculate the complex zeros of the polynomial
    pub fn zeros(&self) -> () {
        ()
    }

    pub fn mod_q(&self,tmpq: &i64) -> Polynomial {
        let new_coeffs: Vec<i64> = self
            .coefficients
            .iter()
            .map(|&coeff| coeff.rem_euclid(*tmpq))
            .collect();
    
        Polynomial::new(new_coeffs)
    }

}



pub fn extended_gcd_poly_mod(f: &Polynomial, g: &Polynomial, phi: &Polynomial) -> (Polynomial, Polynomial, Polynomial) {
    let mut old_r = f.clone();
    let mut r = g.clone();
    let mut old_a = Polynomial::new(vec![1]); // a_0 = 1
    let mut a = Polynomial::new(vec![0]);     // a_1 = 0
    let mut old_b = Polynomial::new(vec![0]); // b_0 = 0
    let mut b = Polynomial::new(vec![1]);     // b_1 = 1


    while !(r.is_zero()) {

        // Ensure the smaller-degree polynomial is on the left
        if old_r.degree() < r.degree() {
            std::mem::swap(&mut old_r, &mut r);
            std::mem::swap(&mut old_a, &mut a);
            std::mem::swap(&mut old_b, &mut b);
        }

        // Get leading coefficients and degrees
        let leading_coeff_old_r = old_r.leading_coefficient();
        let leading_coeff_r = r.leading_coefficient();
        let degree_old_r = old_r.degree();
        let degree_r = r.degree();

        // Calculate the LCM of the leading coefficients
        let lcm = lcm(leading_coeff_old_r, leading_coeff_r);

        // Compute scaling factors
        let scale_old_r = lcm / leading_coeff_old_r;
        let scale_r = lcm / leading_coeff_r;

        // Scale the polynomials
        let scaled_old_r = old_r.multiply_by_scalar(scale_old_r);
        let scaled_r = r.multiply_by_scalar(scale_r);

        // Shift the lower-degree polynomial if necessary
        let degree_diff = degree_old_r - degree_r;
        let shifted_scaled_r = scaled_r.shift(degree_diff);

        // Compute the new remainder
        let new_r = scaled_old_r.delete(&shifted_scaled_r).mod_phi(phi);

        // Update coefficients a and b
        let scaled_a = a.multiply_by_scalar(scale_r).shift(degree_diff);
        let scaled_old_a = old_a.multiply_by_scalar(scale_old_r);
        let new_a = scaled_old_a.delete(&scaled_a).mod_phi(phi);

        let scaled_b = b.multiply_by_scalar(scale_r).shift(degree_diff);
        let scaled_old_b = old_b.multiply_by_scalar(scale_old_r);
        let new_b = scaled_old_b.delete(&scaled_b).mod_phi(phi);

        // Move to the next step
        old_r = r;
        r = new_r;
        old_a = a;
        a = new_a;
        old_b = b;
        b = new_b;
    }

    // At this point, old_r is the GCD, and (old_a, old_b) are the coefficients.
    (old_a, old_b, old_r)
}


pub fn extended_gcd_poly(f: &Polynomial, g: &Polynomial) -> (Polynomial, Polynomial, Polynomial) {
    let mut old_r = f.clone();
    let mut r = g.clone();
    let mut old_a = Polynomial::new(vec![1]); // a_0 = 1
    let mut a = Polynomial::new(vec![0]);     // a_1 = 0
    let mut old_b = Polynomial::new(vec![0]); // b_0 = 0
    let mut b = Polynomial::new(vec![1]);     // b_1 = 1

    while !(r.is_zero()) {
        // Ensure the smaller-degree polynomial is on the left
        if old_r.degree() < r.degree() {
            std::mem::swap(&mut old_r, &mut r);
            std::mem::swap(&mut old_a, &mut a);
            std::mem::swap(&mut old_b, &mut b);
        }

        // Get leading coefficients and degrees
        let leading_coeff_old_r = old_r.leading_coefficient();
        let leading_coeff_r = r.leading_coefficient();
        let degree_old_r = old_r.degree();
        let degree_r = r.degree();

        // Calculate the LCM of the leading coefficients
        let lcm = lcm(leading_coeff_old_r, leading_coeff_r);

        // Compute scaling factors
        let scale_old_r = lcm / leading_coeff_old_r;
        let scale_r = lcm / leading_coeff_r;

        // Scale the polynomials
        let scaled_old_r = old_r.multiply_by_scalar(scale_old_r);
        let scaled_r = r.multiply_by_scalar(scale_r);

        // Shift the lower-degree polynomial if necessary
        let degree_diff = degree_old_r - degree_r;
        let shifted_scaled_r = scaled_r.shift(degree_diff);

        // Compute the new remainder
        let new_r = scaled_old_r.delete(&shifted_scaled_r);

        // Update coefficients a and b
        let scaled_a = a.multiply_by_scalar(scale_r).shift(degree_diff);
        let scaled_old_a = old_a.multiply_by_scalar(scale_old_r);
        let new_a = scaled_old_a.delete(&scaled_a);

        let scaled_b = b.multiply_by_scalar(scale_r).shift(degree_diff);
        let scaled_old_b = old_b.multiply_by_scalar(scale_old_r);
        let new_b = scaled_old_b.delete(&scaled_b);

        // Move to the next step
        old_r = r;
        r = new_r;
        old_a = a;
        a = new_a;
        old_b = b;
        b = new_b;
    }

    // At this point, old_r is the GCD, and (old_a, old_b) are the coefficients.
    (old_a, old_b, old_r)
}

pub fn extended_gcd_poly_example(){
    let phi=Polynomial::new(vec![1,0,0,0,0,0,0,0,0,0,0,0,0,1]);
    let f = Polynomial::new(vec![1, 2,4, 3]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6,0,0,0,7]); // g = 4 + 5x + 6x^2
    let (a, b, gcd) = extended_gcd_poly_mod(&f, &g, &phi);
    let result = a.clone() * f.clone() + b.clone() * g.clone();
    println!("Result: {}", result);
    assert!(result.equal(&gcd, &phi), "GCD is incorrect");

    //Test the case when f is a multiple of g
    let f = Polynomial::new(vec![1, 2,4, 3]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![0,2, 4,8, 6]); // f = 1 + 2x + 3x^2
    let (a, b, gcd) = extended_gcd_poly_mod(&f, &g, &phi);    
    let result = a.clone() * f.clone() + b.clone() * g.clone();
    println!("Result: {}", result);
    assert!(result.equal(&gcd, &phi), "GCD is incorrect");
}





//Test the extended gcd algorithm of polynomial
pub fn extended_gcd_poly_test() {
    // Test the extended GCD algorithm for polynomials
    // Randomly generated two polynomials for 100 times
    let phi=Polynomial::new(vec![1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]);
    for _ in 0..100 {
        //Randomly generate two polynomials f,g
        //The coefficients are randomly generated  
        let f = Polynomial::new((0..8).map(|_| rand::thread_rng().gen_range(0..q)).collect());
        let g = Polynomial::new((0..8).map(|_| rand::thread_rng().gen_range(0..q)).collect());
        let (a, b, gcd) = extended_gcd_poly_mod(&f, &g, &phi);
        let result = a.clone() * f.clone() + b.clone() * g.clone();
        assert!(result.equal(&gcd, &phi), "GCD is incorrect");
        //println!("GCD:{}", gcd);
        println!("Result: {}", result);
    }
}




pub fn test_poly_equal(){
    let phi=Polynomial::new(vec![1,0,0,0,0,0,0,0,0,0,0,0,0,1]);

    let f = Polynomial::new(vec![1, 2, 3]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6]); // g = 4 + 5x + 6x^2
    assert!(!f.equal(&g, &phi), "Polynomial equality is incorrect");

    let h = Polynomial::new(vec![1, 2, 3]); // h = 1 + 2x + 3x^2
    assert!(f.equal(&h, &phi), "Polynomial equality is incorrect");

}




pub fn test_poly_multiplication(){
    let phi=Polynomial::new(vec![1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]);

    let f = Polynomial::new(vec![1, 2, 3]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6]); // g = 4 + 5x + 6x^2
    let h = f * g; // h = f * g = 4 + 13x + 28x^2 + 27x^3 + 18x^4
    let correctanswer=Polynomial::new(vec![4, 13, 28, 27, 18]);
    assert!(h.equal(&correctanswer, &phi), "Polynomial multiplication is incorrect");

    //Generate the second example
    let f = Polynomial::new(vec![3, 2, 3,17, 25]); // f = 3 + 2x + 3x^2 + 17x^3 + 25x^4
    let g = Polynomial::new(vec![4, 5, 6,7,8]); // g = 4 + 5x + 6x^2 + 7x^3 + 8x^4
    let h = f * g; // h = f * g = 12 + 23x + 38x^2 + 101x^3 + 189x^4 + 191x^5 + 189x^6 + 200x^7 + 191x^8 + 200x^9 + 191x^10 + 200x^11 + 191x^12 + 200x^13 + 191x^14 + 200x^15
    let correctanswer=Polynomial::new(vec![12,23,40,116,241,264,293,311,200]);
    assert!(h.equal(&correctanswer, &phi), "Polynomial multiplication is incorrect");
}


pub fn test_poly_deletion(){
    let phi=Polynomial::new(vec![1,0,0,0,0,0,0,0,0,0,0,0,0,1]);

    let f = Polynomial::new(vec![1, 2, 3]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6]); // g = 4 + 5x + 6x^2
    let h = f.delete(&g); // h = f - g = -3 - 3x - 3x^2
    let correctanswer=Polynomial::new(vec![1-4, 2-5, 3-6]);
    assert!(h.equal(&correctanswer, &phi), "Polynomial deletion is incorrect");

    //Generate the second example
    //Now f has larger order than g
    let f = Polynomial::new(vec![3, 2, 3,17, 25,27,29]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6,7,8]); // g = 4 + 5x + 6x^2
    let h = f.delete(&g); // h = f - g = -1 - 3x - 3x^2
    let correctanswer=Polynomial::new(vec![3-4, 2-5, 3-6,17-7,25-8,27,29]);
    assert!(h.equal(&correctanswer, &phi), "Polynomial deletion is incorrect");

    //Generate the third example
    //Now g has larger order than f
    let g = Polynomial::new(vec![3, 2, 3,17, 25,27,29]); // f = 1 + 2x + 3x^2
    let f = Polynomial::new(vec![4, 5, 6,7,8]); // g = 4 + 5x + 6x^2
    let h = f.delete(&g); // h = f - g = -1 - 3x - 3x^2
    let correctanswer=Polynomial::new(vec![4-3, 5-2, 6-3,7-17,8-25,-27,-29]);
    assert!(h.equal(&correctanswer, &phi), "Polynomial deletion is incorrect");

}




pub fn test_poly_sum(){
    let phi=Polynomial::new(vec![1,0,0,0,0,0,0,0,0,0,0,0,0,1]);

    let f = Polynomial::new(vec![1, 2, 3]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6]); // g = 4 + 5x + 6x^2
    let h = f+g; // h = f + g = 5 + 7x + 9x^2
    let correctanswer=Polynomial::new(vec![5, 7, 9]);
    assert!(h.equal(&correctanswer, &phi), "Polynomial multiplication is incorrect");

    //Generate the second example
    let f = Polynomial::new(vec![3, 2, 3,17, 25]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6,7,8]); // g = 4 + 5x + 6x^2
    let h = f+g; // h = f + g = 7 + 7x + 9x^2 + 24x^3 + 33x^4
    let correctanswer=Polynomial::new(vec![7, 7, 9, 24, 33]);
    assert!(h.equal(&correctanswer, &phi), "Polynomial multiplication is incorrect");
}

pub fn test_leading_coeff(){
    let f = Polynomial::new(vec![1, 2, 1, 3]); // f = 1 + 2x + 3x^2
    let g = Polynomial::new(vec![4, 5, 6]); // g = 4 + 5x + 6x^2
    assert_eq!(f.leading_coefficient(),3);
    assert_eq!(g.leading_coefficient(),6);
    let f = Polynomial::new(vec![1, 2, 1, 30,0,0]); // f = 1 + 2x + 3x^2
    assert_eq!(f.leading_coefficient(),30);
    let f =Polynomial::new(vec![0]); // f = 1 + 2x + 3x^2
    assert_eq!(f.leading_coefficient(),0);
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


//We will use fieldnorm to NTRU equation onto a smaller ring
pub fn fieldnorm(poly: &Polynomial) -> Polynomial {
    Polynomial::new(poly.coefficients.clone())
}



pub fn test_poly_example(){
 
    let tmpq=12289 as i64;
    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1]); // φ = x^8 + 1
    let f=Polynomial::new(vec![-55,11,-23,-23,47,16,13,61]); //f
    let g=Polynomial::new(vec![-25,-24,30,-3,36,-39,6]); //g
    let h=Polynomial::new(vec![-4839,-6036,-4459,-2665,-186,-4303,3388,-3568]); //h

    //verify that h=g*finv mode phi mode q
    let leftg=(h.clone()*f).mod_phi(&phi).mod_q(&tmpq);
    assert!(leftg.equal_exact(&g), "h=g*finv mod phi mod q is incorrect");
}



pub fn test_poly_inverse_example(){
 
    let tmpq=12289 as i64;
    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1]); // φ = x^8 + 1
    let f=Polynomial::new(vec![-55,11,-23,-23,47,16,13,61]); //f
    let g=Polynomial::new(vec![-25,-24,30,-3,36,-39,6]); //g
    let h=Polynomial::new(vec![-4839,-6036,-4459,-2665,-186,-4303,3388,-3568]); //h

    let finverse=f.inverse(&phi);

    let result=(g*finverse).mod_phi(&phi).mod_q(&tmpq);

    assert!(result.equal_exact(&h), "Polynomial inverse is incorrect");
}

pub fn test_poly_inverse(){
    let tmpq=12289 as i64;
    let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0,0,0,0,0,0,0,0,0,0,1]); // φ = x^8 + 1
    //Randomly generate 100 f, test if the inverse is correct
    for _ in 0..100{
        let f=Polynomial::new((0..8).map(|_| rand::thread_rng().gen_range(0..q)).collect());
        let finverse=f.inverse(&phi);
        let result=(f.clone()*finverse).mod_phi(&phi).mod_q(&tmpq);
        assert!(result.equal_exact(&Polynomial::new(vec![1])), "Polynomial inverse is incorrect");
    }
}



//Calculate all the unit roots of a given order n
//The equation is x^n=1
pub fn calculate_unit_roots(n: &i64) -> Vec<(f64, f64)> {
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
    let unit_roots=calculate_unit_roots(&(phideg as i64));
    for i in 0..phideg{
        let root=unit_roots[i];
        let mut sum=Complex::new(0.0,0.0);
        let xi=Complex::new(root.0,root.1);
        for j in 0..phideg{
            if j>=poly.coefficients.len(){
                break;
            }
            let aj=Complex::new(poly.coefficients[j] as f64,0.0);
            sum+=aj*xi.powf(j as f64);
        }
        fft[i]=sum.re;
        fft[i+phideg]=sum.im;
    }
    println!("{:?}",fft.len());
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


//Use the input fft vector to calculate the polynomial coefficients
pub fn inverseFFT(phi: &Polynomial, fft: &Vec<f64>)->Polynomial{
    let phideg=phi.degree();
    let mut coefficients=vec![0 as i64;phideg];
    let unit_roots=calculate_unit_roots(&(phideg as i64));
    for i in 0..phideg{
        let mut sum=Complex::new(0.0,0.0);
        for j in 0..phideg{
            //xj in the van der monde matrix
            //yi in the fft representation
            let tmpcomplex=Complex::new(fft[j],fft[j+phideg]);
            let root=Complex::new(unit_roots[j].0,unit_roots[j].1);
            sum+=tmpcomplex*root.powf(-(i as f64));
        }
        if phideg>1{
            coefficients[i]=(sum.re as i64 /(phideg as i64 -1)) as i64;
        }
        else{
            coefficients[i]=sum.re as i64;
        }
    }
    Polynomial::new(coefficients)
}



//The NTT representation of a polynomial modulo phi
pub fn NTT(poly: &Polynomial, phi: &Polynomial)-> Vec<i64>{
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
