from Reference.Algorithm_2 import X


class BlindRotateCore:
    pass


def RGSW(param, modulus):
    pass


def BlindRotateRoundToOdd(f, alpha, beta, brki, akgu, ak_minus_g):
    """
    Procedure for blind rotation of a ciphertext encrypting a univariate polynomial using the Automorphism technique in Homomorphic Encryption.

    Parameters:
    f (Polynomial): a univariate polynomial
    alpha (list): a vector of odd scalars'
    beta (int): an odd scalar
    brki (dict): a set of blind rotations for each variable of the polynomial
    akgu (list): a set of automorphism keys
    ak_minus_g (RGSW): an automorphism key

    Returns:
    RGSW: a ciphertext encrypting the product of f(X), X^β, and the blind rotations {brki}
    """
    acc = RGSW(0, f.modulus)
    X_minus_g = [X[i] for i in range(len(X)) if i != len(X) - 1]
    acc = acc.add(f.mult(X_minus_g, -1).mult(X[len(X) - 1], beta))
    acc = BlindRotateCore(acc, alpha, brki, akgu, ak_minus_g)
    return acc
