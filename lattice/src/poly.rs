use std::fmt;
use std::ops::{Add, Mul, Rem};

/// Represents a polynomial over Z_q[x]
#[derive(Debug, Clone)]
pub struct Polynomial {
    coefficients: Vec<i32>, // Coefficients of the polynomial
    q: i32,                // Modulus
}

impl Polynomial {
    /// Creates a new polynomial with given coefficients and modulus
    pub fn new(coefficients: Vec<i32>, q: i32) -> Self {
        let coefficients = coefficients.into_iter().map(|c| c.rem_euclid(q)).collect();
        Polynomial { coefficients, q }
    }


    /// Degree of the polynomial
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Shifts a polynomial by a given number of degrees (multiply by x^degree),
    pub fn shift(&self, degree: usize)-> Polynomial{
        // Create a shifted polynomial by appending `degree` zeros
        let mut new_coeffs = vec![0; degree];
        new_coeffs.extend(&self.coefficients);

        // Create a new Polynomial object
        let shifted_poly = Polynomial::new(new_coeffs, self.q);
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
                new_coeffs[index]=(self_coeffs[index]-other_coeffs[index]).rem_euclid(self.q);
        }    
        let mut newpoly=Polynomial::new(new_coeffs, self.q);
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
        let selflast = *selfcoeffts.first().unwrap(); // Extract the value and dereference it
        let newpoly=self.delete(&phi.shift(gap).multiple(selflast));
        println!("Newpoly");
        println!("{}",newpoly);
        // newpoly.mod_phi(phi)
        self.clone()
    }
    
    
    
    pub fn to_matrix(&self, phi: &Polynomial) -> Vec<Vec<i32>> {
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

    pub fn multiple(&self, value: i32) -> Polynomial {
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
