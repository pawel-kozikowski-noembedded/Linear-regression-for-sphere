import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go 
import re

def wczytaj_dane_magnetometru(nazwa_pliku: str) -> np.ndarray | None:
    punkty = []
    pattern = re.compile(r"type=DATA_MAG.*x=([\d-]+).*y=([\d-]+).*z=([\d-]+)")

    try:
        with open(nazwa_pliku, 'r') as f:
            for linia in f:
                match = pattern.search(linia)
                if match:
                    x, y, z = map(int, match.groups())
                    punkty.append([x, y, z])

        if not punkty:
            print("Nie znaleziono żadnych pasujących danych w pliku.")
            return None

        return np.array(punkty, dtype=np.float64)
    except FileNotFoundError:
        print(f"Błąd: Plik '{nazwa_pliku}' nie został znaleziony.")
        return None

def solve_linear_system(M, R):
    try:
        return np.linalg.solve(M, R)
    except np.linalg.LinAlgError:
        return None


def dopasuj_sfere(points: np.ndarray) -> dict:
    srodek_przyblizony = np.mean(points, axis=0)
    dane_scentrowane = points - srodek_przyblizony

    N = len(dane_scentrowane)
    x = dane_scentrowane[:, 0]
    y = dane_scentrowane[:, 1]
    z = dane_scentrowane[:, 2]

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_z = np.sum(z)

    sum_x2 = np.sum(x ** 2)
    sum_y2 = np.sum(y ** 2)
    sum_z2 = np.sum(z ** 2)

    sum_xy = np.sum(x * y)
    sum_xz = np.sum(x * z)
    sum_yz = np.sum(y * z)

    f = x ** 2 + y ** 2 + z ** 2
    sum_xf = np.sum(x * f)
    sum_yf = np.sum(y * f)
    sum_zf = np.sum(z * f)
    sum_f = np.sum(f)

    M = np.array([
        [2 * sum_x2, 2 * sum_xy, 2 * sum_xz, sum_x],
        [2 * sum_xy, 2 * sum_y2, 2 * sum_yz, sum_y],
        [2 * sum_xz, 2 * sum_yz, 2 * sum_z2, sum_z],
        [sum_x, sum_y, sum_z, N]
    ])

    R = np.array([sum_xf, sum_yf, sum_zf, sum_f])

    v = solve_linear_system(M, R)

    xc_scentrowany, yc_scentrowany, zc_scentrowany, k = v

    xc_final = xc_scentrowany + srodek_przyblizony[0]
    yc_final = yc_scentrowany + srodek_przyblizony[1]
    zc_final = zc_scentrowany + srodek_przyblizony[2]

    promien = np.sqrt(k + xc_scentrowany ** 2 + yc_scentrowany ** 2 + zc_scentrowany ** 2)

    return {
        "offset_x": xc_final,
        "offset_y": yc_final,
        "offset_z": zc_final,
        "promien": promien
    }

def przesun_punkty(punkty: np.ndarray, offset: np.array) -> None:
    for punkt in punkty:
        punkt -= offset
    
    return punkty

def wykres_3D(dane: np.ndarray) -> None:
    x_data: list[np.float64] = [punkt[0] for punkt in dane]
    y_data: list[np.float64] = [punkt[1] for punkt in dane]
    z_data: list[np.float64] = [punkt[2] for punkt in dane]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data, y_data, z_data, c='b', marker=',')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') 
    
    plt.savefig("3d.png")
    
def wykres_3D_interactive(dane: np.ndarray, promien: float) -> None:
    # a scatter plot for the data points
    scatter = go.Scatter3d(
        x=dane[:,0], y=dane[:,1], z=dane[:,2],
        mode='markers',
        marker=dict(size=4, color=dane[:,2], colorscale='Viridis', opacity=0.8),
        name='Pomiary'
    )

    # create sphere vertices
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_s = promien * np.outer(np.cos(u), np.sin(v))
    y_s = promien * np.outer(np.sin(u), np.sin(v))
    z_s = promien * np.outer(np.ones(np.size(u)), np.cos(v))

    sphere = go.Surface(
        x=x_s, y=y_s, z=z_s,
        colorscale=[[0, 'red'], [1, 'red']],
        opacity=0.3,
        showscale=False,
        name='Dopasowana sfera'
    )

    fig = go.Figure(data=[scatter, sphere])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.show()

dane = wczytaj_dane_magnetometru("data.txt")
wyniki = dopasuj_sfere(dane)

print(wyniki["offset_x"])
print(wyniki["offset_y"])
print(wyniki["offset_z"])
print(wyniki["promien"])
