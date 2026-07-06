template_program = '''
def approximate(x: float) -> float:
    """Approximate sin(x) for x in [-pi, pi] without calling a sine function."""
    return x
'''

task_description = (
    "Approximate sin(x) on [-pi, pi] without directly calling a standard sine "
    "implementation such as math.sin, numpy.sin, or equivalent imports. The "
    "score rewards low RMSE and low maximum absolute error on a fixed set of "
    "sample points."
)
