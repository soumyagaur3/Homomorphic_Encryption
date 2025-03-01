def BlindRotateOptim(f, alpha, beta, brki, brknsum, akgu, ak_minus_g):
    """
    Procedure for blind rotation of a ciphertext encrypting a univariate polynomial using the Automorphism technique in Homomorphic Encryption.

    Parameters:
    f (Polynomial): a univariate polynomial
    alpha (list): a vector of scalars
    beta (int): a scalar
    brki (dict): a set of blind rotations for each variable of the polynomial
    brknsum (RGSW): a blindrotation for the sum of all variables
akgu (list): a set of automorphism keys
    ak_minus_g (RGSW): an automorphism key

    Returns:
    RGSW: a ciphertext encrypting the product of f(X), X2(β+⟨α,s⟩), and the blind rotations {brki}
    """
    acc = RGSWz(0, f.modulus)
    X_minus_g = [X[i] for i in range(len(X)) if i != len(X) - 1]
    acc = acc.add(f.mult(X_minus_g, -2).mult(X[len(X) - 1], beta))
    acc = acc.hom_mult(brknsum)
    alpha_prime = [(2 * alpha_i + 1) % (2 * N) for alpha_i in alpha]
    acc = BlindRotateCore(acc, alpha_prime, brki, akgu, ak_minus_g)
    return acc