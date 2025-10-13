import pathlib

from math import sqrt
from parse import load_system

A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")

def determinant(matrix: list[list[float]]) -> float:
    if len(matrix)==2 :
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    det=matrix[0][0]*matrix[1][1]*matrix[2][2]
    det=det+matrix[1][0]*matrix[2][1]*matrix[0][2]
    det=det+matrix[2][0]*matrix[0][1]*matrix[1][2]
    det=det-matrix[0][2]*matrix[1][1]*matrix[2][0]
    det=det-matrix[1][2]*matrix[2][1]*matrix[0][0]
    det=det-matrix[2][2]*matrix[0][1]*matrix[1][0]
    return det

print(f"{determinant(A)=}")

def trace(matrix: list[list[float]]) -> float:
    trace=0
    for i in range(len(matrix)):
        trace+=matrix[i][i]
    return trace

print(f"{trace(A)=}")

def norm(vector: list[float]) -> float:
    sum=0.0
    for x in vector:
        sum+=x*x
    return sqrt(sum)

print(f"{norm(B)=}")

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    transpose: list[list[float]]=[]
    for i in range(len(matrix[0])):
        line:list[float]=[]
        for j in range(len(matrix)):
            line.append(matrix[j][i])
        transpose.append(line)
    return transpose

print(f"{transpose(A)=}")

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result: list[float]=[]
    for i in range(len(matrix)):
        sum=0
        for j in range(len(matrix[0])):
            sum+=matrix[i][j]*vector[j]
        result.append(sum)
    return result

print(f"{multiply(A, B)=}")

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    minor: list[list[float]]=[]
    for k in range(len(matrix)):
        row: list[float]=[]
        for l in range(len(matrix[0])):
            if k!=i and l!=j:
                row.append(matrix[k][l])
        if len(row)>0:
            minor.append(row)
    return minor
print(f"{minor(A, 1, 2)=}")

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cof: list[list[float]]=[]
    n=len(matrix)
    for i in range(n):
        row: list[float]=[]
        for j in range(n):
            det=determinant(minor(matrix, i, j))
            if (i+j)%2==1:
                det=-det
            row.append(det)
        cof.append(row)
    return cof

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))

def inverse(matrix: list[list[float]]) -> list[list[float]]:
    adj=adjoint(matrix)
    det=determinant(matrix)
    if det==0:
        return []
    for i in range(len(adj)):
        for j in range(len(adj)):
            adj[i][j]=adj[i][j]/det
    return adj