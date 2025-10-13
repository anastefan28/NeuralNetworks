import pathlib
from operations import adjoint, determinant, inverse, multiply
from parse import load_system

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det=determinant(matrix)
    if det==0:
        return []
    results: list[float]=[]
    for i in range(len(matrix)):
        temp=[]
        for row in matrix:
            new_row=row.copy()     
            temp.append(new_row)
        for j in range(len(matrix)):
            temp[j][i]=vector[j]
        results.append(determinant(temp)/det)
    return results

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inv=inverse(matrix)
    return multiply(inv, vector)
    
A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")
print(f"{solve_cramer(A, B)=}")
print(f"{solve(A, B)=}")