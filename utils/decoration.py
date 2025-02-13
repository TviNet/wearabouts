# This function adds a comment to each line of Python code
def comment_lines(code: str) -> str:
    # Split the code into individual lines
    lines = code.splitlines()

    # Add a comment character to the start of each non-empty line
    commented_lines = []
    for line in lines:
        # Skip empty lines
        if line.strip():
            commented_lines.append(f"# {line}")
        else:
            commented_lines.append(line)

    # Join the lines back together
    return "\n".join(commented_lines)


def indent_lines(code: str, spaces: int = 4) -> str:
    # Split the code into individual lines
    lines = code.splitlines()

    # Add indentation to each line
    indented_lines = []
    indent = " " * spaces
    for line in lines:
        if line.strip():
            indented_lines.append(f"{indent}{line}")
        else:
            indented_lines.append(line)

    # Join the lines back together
    return "\n".join(indented_lines)
