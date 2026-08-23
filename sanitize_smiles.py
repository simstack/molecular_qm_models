
def sanitize_smiles_for_filename(smiles: str, max_length: int = 200) -> str:
    """Return a filesystem-friendly token derived from a SMILES string."""
    if not smiles:
        return "nosmiles"
    substitutions = {
        "(": "_LBR_",
        ")": "_RBR_",
        "[": "_LSQB_",
        "]": "_RSQB_",
        "/": "_SLASH_",
        "\\": "_BSLASH_",
        "#": "_TB_",
        "=": "_DB_",
        "@": "_AT_",
        "+": "_PLUS_",
        "-": "_MINUS_",
        ".": "_DOT_",
    }
    safe = smiles
    for char, sub in substitutions.items():
        if char in safe:
            safe = safe.replace(char, sub)
    if not safe:
        return "nosmiles"
    if len(safe) > max_length:
        safe = safe[: max_length - 10] + "_TRUNC"
    return safe

