use rand::Rng;

//The extended gcd algorithm for integers. Input a,b
//Return s,t,gcd(a,b), such that sa+tb=gcd(a,b)
pub fn extended_gcd_integer(a: i64, b: i64) -> (i64, i64, i64) {
    let mut old_r = a;
    let mut r = b;
    let mut old_s = 1;
    let mut s = 0;
    let mut old_t = 0;
    let mut t = 1;
    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;

        let temp_t = t;
        t = old_t - quotient * t;
        old_t = temp_t;
    }
    (old_s, old_t, old_r)
}

pub fn extended_gcd_integer_test() {
    // Test the extended GCD algorithm for integers
    // Randomly generated two large numbers for 100 times
    for _ in 0..100 {
        let a = rand::thread_rng().gen_range(1..1000);
        let b = rand::thread_rng().gen_range(1..1000);
        let (s, t, gcd) = extended_gcd_integer(a, b);
        assert_eq!(s * a + t * b, gcd, "GCD is incorrect");
        println!("GCD of {} and {}: {}, s = {}, t = {}", a, b, gcd, s, t);
    }
}


// Helper function to compute the least common multiple (LCM) of two integers
pub fn lcm(a: i64, b: i64) -> i64 {
    (a * b).abs() / gcd(a, b)
}

// Helper function to compute the greatest common divisor (GCD) of two integers
pub fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

//Calculate the inverse ainversextended_gcd_ine of a modulo m, such that a*ainverse=1 mod m
pub fn inverse_mod(a: i64, m: i64) -> i64 {
    let (inv, _, _) = extended_gcd_integer(a, m);
    inv.rem_euclid(m)
}


pub fn test_inverse_mod(){
    let a=3;
    let m=7;
    let inv=inverse_mod(a,m);
    println!("The inverse of {} modulo {} is {}",a,m,inv);
    assert_eq!(a*inv%m,1);
}

