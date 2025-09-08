def cantor_pair(a, b):
    """Cantor pairing function to uniquely encode two non-negative integers into one.
    Args:
        a (int): First non-negative integer.
        b (int): Second non-negative integer.
    Returns:
        int: A unique non-negative integer representing the pair (a, b).
    """
    return (a + b) * (a + b + 1) // 2 + b

