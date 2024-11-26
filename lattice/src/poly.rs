use std::fmt;
use std::ops::{Add, Mul, Sub, Rem};
use std::default::Default;


/// Represents a polynomial over Z_q[x]
#[derive(Debug, Clone)]
pub struct Polynomial {
    coefficients: Vec<i32>, // Coefficients of the polynomial
    q: i32,                // Modulus
}

impl Polynomial {
    /// Creates a new polynomial with given coefficients and modulus
    /// The coefficients are reduced modulo q, i.e., coefficients[i] = coefficients[i] % q
    /// The coefficients are put in rising order of degrees. For example, [1, 2, 3] represents 1 + 2x + 3x^2
    pub fn new(coefficients: Vec<i32>, q: i32) -> Self {
        let coefficients = coefficients.into_iter().map(|c| c.rem_euclid(q)).collect();
        Polynomial { coefficients, q }
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
    
            println!(
                "x^{} * f = {:?} mod φ = {:?}",
                i, shifted_poly.coefficients, reduced_poly.coefficients
            );

            println!(
                "x^{} * f = {} mod φ = {}",
                i, &shifted_poly, &reduced_poly
            );
    
            // Fill the row with the reduced coefficients
            for (j, &coeff) in reduced_poly.coefficients.iter().enumerate() {
                matrix[j][i] = coeff;
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





//Operator overloading for subtraction, p1*p2
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
            newcoeffs[i]=(*selfval+*rhsval).rem_euclid(self.q);
        }
        Polynomial::new(newcoeffs,self.q)
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
            newcoeffs[i]=(*selfval-*rhsval).rem_euclid(self.q);
        }
        Polynomial::new(newcoeffs,self.q)
    }
}



impl Default for Polynomial {
    fn default() -> Self {
        Polynomial {
            coefficients: vec![],
            q: 0,
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*; // Import all items from the parent module

    #[test]
    fn test_polynomial_to_matrix() {
        // Define the modulus
        let phi = Polynomial::new(vec![1, 0, 0, 0, 1], 5); // φ = x^4 + 1

        // Define the polynomial f = x^2 + 1
        let f = Polynomial::new(vec![1, 0, 1], 5); // Coefficients: [0, 0, 1, 0, 1]

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



    #[test]
    fn test_multiplication(){
        let phi = Polynomial::new(vec![1, 0, 0, 0,0 ,0,0,0, 1], 12289); // φ = x^8 + 1
        let f=Polynomial::new(vec![-55,11,-23,-23,47,16,13,61],12289); //f
        let g=Polynomial::new(vec![-25,-24,30,-3,36,-39,6],12289); //g
        let F=Polynomial::new(vec![58,20,17,-64,-3,-9,-21,-84],12289); //G
        let G=Polynomial::new(vec![-41,-34,-33,25,-41,31,-18,-32],12289); //G
        let h=Polynomial::new(vec![-4839,-6036,-4459,-2665,-186,-4303,3388,-3568],12289); //h





    }



}
