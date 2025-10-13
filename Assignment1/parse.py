import pathlib

def extract_coefficient(idx: int, parts: list[str]) -> float:
    sign=1.0
    if idx>0 and parts[idx-1]=='-':
        sign=-1.0
    term=parts[idx]
    number=term[:-1]
    if number=='':
        coef=1.0
    else:
        coef=float(number)
    return sign*coef

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A: list[list[float]] = []
    B: list[float] = []
    with path.open("r") as f:
        for line in f:
            line=line.strip()
            left=line.split("=")[0].strip()
            right=line.split("=")[1].strip()
            terms = left.split()                 
            A.append([extract_coefficient(0, terms),
                       extract_coefficient(2, terms), 
                       extract_coefficient(4, terms)])
            B.append(float(right))
    return A, B

