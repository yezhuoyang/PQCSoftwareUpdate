# PQCSoftwareUpdate
We study the software update scheme with post quantum signature.



# How to sign your commit?

Since we are working at signing software update, we should also sign each of our commit.
Please generate a GPG key pairs:

https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key

And then set the following configuration for your local git:

```bash
git config --local commit.gpgsign true
gpg --list-secret-keys --keyid-format LONG
```

After you get you long secret key, config it as:

```bash
git config --local user.signingkey "[GPG_KEY]"
```

In windows, you might need to configure your gpg path:
```
where.exe gpg
git config --global gpg.program [PATH_HERE]
```


# Compile and run tests

The project is a rust implementation of falcon.
To compile and run the project, execute:

```bash
cd lattice
cargo run
```

To run the tests:

```bash
cargo test
```


# Implementation

Polynomial class: A class representing an integer polynomial

- [x] Shift
- [x] to_matrix
- [x] Modular phi
- [ ] Multiplication
- [ ] Addition
- [x] Minus
- [ ] Inverse

Hash function with shake256

- [x] Hash any binary function with shake256
- [ ] Map a Hashed string to a lattice vector

NTRU class: A class that store the NTRU publickey and private key

- [ ] Generate f,g by Gaussian sampling
- [ ] Verify the validity of f,g
- [ ] Solve for F,G
- [ ] Calculate h
- [ ] Generate signature for any file 
- [ ] Verify signature with public key


TLS with Falcon

- [ ] TLS protocal


Benchmark

- [ ] Benchmark memory
- [ ] Benchmark time


# Reference

https://falcon-sign.info/




