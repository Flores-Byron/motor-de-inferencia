import re
import unicodedata

import pandas as pd
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# 1. FÓRMULAS (árbol sintáctico con tuplas)
#    ('atomo', 'p') | ('no', f) | ('y', a, b) | ('o', a, b) | ('si', a, b) | ('sii', a, b)
# ----------------------------------------------------------------------------
SIMBOLOS = {"y": "∧", "o": "∨", "si": "→", "sii": "↔️"}
PRIORIDAD = {"sii": 1, "si": 2, "o": 3, "y": 4, "no": 5, "atomo": 6}


def a_texto(f):
    """Convierte una fórmula a texto, con paréntesis solo cuando hacen falta."""
    tipo = f[0]
    if tipo == "atomo":
        return f[1]
    if tipo == "no":
        hijo = f[1]
        if hijo[0] in ("atomo", "no"):
            return "¬" + a_texto(hijo)
        return "¬(" + a_texto(hijo) + ")"

    def envolver(h):
        if h[0] in ("atomo", "no"):
            return a_texto(h)
        if h[0] == tipo and tipo in ("y", "o"):  # asociativos: p ∧ q ∧ r
            return a_texto(h)
        return "(" + a_texto(h) + ")"  # cualquier otra combinación: paréntesis explícitos

    return f"{envolver(f[1])} {SIMBOLOS[tipo]} {envolver(f[2])}"


def evaluar(f, asignacion):
    tipo = f[0]
    if tipo == "atomo":
        return asignacion.get(f[1], False)
    if tipo == "no":
        return not evaluar(f[1], asignacion)
    a, b = evaluar(f[1], asignacion), evaluar(f[2], asignacion)
    return {"y": a and b, "o": a or b, "si": (not a) or b, "sii": a == b}[tipo]


# ----------------------------------------------------------------------------
# 2. ANALIZADOR DE LENGUAJE NATURAL
# ----------------------------------------------------------------------------
LETRAS = list("abcdefghijklmnopqrstuvwxyz")


def sin_acentos(texto):
    """Quita acentos carácter por carácter (conserva la longitud)."""
    salida = []
    for c in texto:
        base = "".join(x for x in unicodedata.normalize("NFD", c) if not unicodedata.combining(x))
        salida.append(base[0] if base else c)
    return "".join(salida)


def limpiar(texto):
    texto = texto.strip().strip(".,;:!¡?¿").strip()
    return re.sub(r"\s+", " ", texto)


class TablaAtomos:
    """se ignoran mayúsculas, acentos y puntuación."""

    def __init__(self):
        self.por_clave = {}  # clave -> letra
        self.significado = {}  # letra -> texto original

    def obtener(self, texto):
        texto = limpiar(texto)
        if not texto:
            raise ValueError("Hay una proposición que esta vacía.")
        clave = sin_acentos(texto).lower()
        if clave not in self.por_clave:
            n = len(self.por_clave)
            letra = LETRAS[n] if n < len(LETRAS) else f"p{n - len(LETRAS) + 1}"
            self.por_clave[clave] = letra
            self.significado[letra] = texto[0].upper() + texto[1:]
        return self.por_clave[clave]


PAT_SII = r"\s+(?:si\s+y\s+(?:solo|solamente)\s+si|sii)\s+"
PAT_SI = r"^(?:si\s+)?(.+?)\s*,?\s+entonces\s+(.+)$"
PAT_NO_CIERTO = r"^no\s+es\s+(?:cierto|verdad|el\s+caso)\s+que\s+(.+)$"


def _dividir(texto, plano, patron):
    """Divide texto en las posiciones donde patron coincide en plano."""
    partes, inicio = [], 0
    for m in re.finditer(patron, plano):
        partes.append(texto[inicio:m.start()])
        inicio = m.end()
    partes.append(texto[inicio:])
    return partes


def _unir(tipo, formulas):
    resultado = formulas[0]
    for f in formulas[1:]:
        resultado = (tipo, resultado, f)
    return resultado


def analizar(texto, atomos):
    """Convierte una oración en español a fórmula. Precedencia (de menor a mayor):
    «si y solo si» < «si… entonces» < «o» < «y» < «no»."""
    texto = limpiar(texto)
    plano = sin_acentos(texto).lower()

    # «si y solo si»  ->  ↔️
    partes = _dividir(texto, plano, PAT_SII)
    if len(partes) == 2:
        return ("sii", analizar(partes[0], atomos), analizar(partes[1], atomos))
    if len(partes) > 2:
        raise ValueError(f"Demasiados «si y solo si» en una misma regla: «{texto}»")

    # «si A entonces B»  ->  →
    m = re.match(PAT_SI, plano)
    if m:
        a = texto[m.start(1):m.end(1)]
        b = texto[m.start(2):m.end(2)]
        return ("si", analizar(a, atomos), analizar(b, atomos))
    if re.match(r"^si\s", plano) and "entonces" not in plano:
        raise ValueError(f"Falta «entonces» en: «{texto}»")

    # «no es cierto que ...»  ->  ¬ de toda la oración
    m = re.match(PAT_NO_CIERTO, plano)
    if m:
        return ("no", analizar(texto[m.start(1):m.end(1)], atomos))

    # «o»  ->  ∨
    partes = _dividir(texto, plano, r"\s+o\s+")
    if len(partes) > 1:
        return _unir("o", [analizar(p, atomos) for p in partes])

    # «y»  ->  ∧
    partes = _dividir(texto, plano, r"\s+y\s+")
    if len(partes) > 1:
        return _unir("y", [analizar(p, atomos) for p in partes])

    # «no ...»  ->  ¬
    if re.match(r"^no\s+\S", plano):
        return ("no", analizar(texto[3:], atomos))

    return ("atomo", atomos.obtener(texto))


def separar_reglas(texto):
    """Una regla por línea (también se aceptan ';' o '.' como separador)."""
    reglas = []
    for linea in re.split(r"[\n;]+", texto):
        for trozo in re.split(r"\.(?:\s+|$)", linea):
            if trozo.strip():
                reglas.append(trozo.strip())
    return reglas


# ----------------------------------------------------------------------------
# 3. TABLERO SEMÁNTICO
# ----------------------------------------------------------------------------
class Nodo:
    def __init__(self, idx, formula, padre=None, etiqueta=""):
        self.idx = idx
        self.formula = formula
        self.padre = padre
        self.hijos = []
        self.etiqueta = etiqueta  # "R1", "R2"... solo en la base de conocimiento
        self.aplicada = None  # número de paso en el que se desarrolló (✓)
        self.cerrada = False  # hoja cerrada (✕)
        self.contradiccion = None


def es_literal(f):
    return f[0] == "atomo" or (f[0] == "no" and f[1][0] == "atomo")


def neg(f):
    return ("no", f)


def clasificar(f):
    """Devuelve (tipo, nombre_regla, ramas). Cada rama es una lista de fórmulas en cadena.
    alfa: una sola rama | beta: varias ramas. None si es literal."""
    if es_literal(f):
        return None
    if f[0] == "y":
        return "alfa", "∧", [[f[1], f[2]]]
    if f[0] == "o":
        return "beta", "∨", [[f[1]], [f[2]]]
    if f[0] == "si":
        return "beta", "→", [[neg(f[1])], [f[2]]]
    if f[0] == "sii":
        return "beta", "↔️", [[("y", f[1], f[2])], [("y", neg(f[1]), neg(f[2]))]]
    # f = ¬g con g compuesta
    g = f[1]
    if g[0] == "no":
        return "alfa", "¬¬", [[g[1]]]
    if g[0] == "o":
        return "alfa", "¬∨", [[neg(g[1]), neg(g[2])]]
    if g[0] == "si":
        return "alfa", "¬→", [[g[1], neg(g[2])]]
    if g[0] == "y":
        return "beta", "¬∧", [[neg(g[1])], [neg(g[2])]]
    if g[0] == "sii":
        return "beta", "¬↔️", [[("y", g[1], neg(g[2]))], [("y", neg(g[1]), g[2])]]
    raise ValueError("Fórmula desconocida")


class Tablero:
    MAX_NODOS = 1500

    def __init__(self, formulas, limite_pasos=None):
        self.nodos = []
        self.pasos = []
        self.truncado = False
        self.completo = False
        self.raiz = None

        ultimo = None
        for i, f in enumerate(formulas):
            ultimo = self._nuevo(f, ultimo, etiqueta=f"R{i + 1}")
        self._revisar_cierre(ultimo)

        while True:
            if limite_pasos is not None and len(self.pasos) >= limite_pasos:
                break
            if len(self.nodos) > self.MAX_NODOS:
                self.truncado = True
                break
            elegido = self._elegir()
            if elegido is None:
                self.completo = True
                break
            self._aplicar(elegido)

    # --- construcción ---------------------------------------------------
    def _nuevo(self, formula, padre, etiqueta=""):
        nodo = Nodo(len(self.nodos) + 1, formula, padre, etiqueta)
        self.nodos.append(nodo)
        if padre is None:
            self.raiz = nodo
        else:
            padre.hijos.append(nodo)
        return nodo

    def _hojas_abiertas(self, nodo):
        if not nodo.hijos:
            return [] if nodo.cerrada else [nodo]
        hojas = []
        for h in nodo.hijos:
            hojas += self._hojas_abiertas(h)
        return hojas

    def _revisar_cierre(self, hoja):
        positivos, negativos = set(), set()
        n = hoja
        while n is not None:
            f = n.formula
            if f[0] == "atomo":
                positivos.add(f[1])
            elif f[0] == "no" and f[1][0] == "atomo":
                negativos.add(f[1][1])
            n = n.padre
        comunes = positivos & negativos
        if comunes:
            hoja.cerrada = True
            hoja.contradiccion = sorted(comunes)[0]

    def _elegir(self):
        """Primero reglas α (no ramifican), luego β; dentro de cada tipo, de arriba hacia abajo."""
        mejor = None
        for nodo in self.nodos:
            if nodo.aplicada is not None or es_literal(nodo.formula):
                continue
            if not self._hojas_abiertas(nodo):
                continue
            tipo = clasificar(nodo.formula)[0]
            if tipo == "alfa":
                return nodo
            if mejor is None:
                mejor = nodo
        return mejor

    def _aplicar(self, nodo):
        tipo, regla, ramas = clasificar(nodo.formula)
        paso = len(self.pasos) + 1
        hojas = self._hojas_abiertas(nodo)
        for hoja in hojas:
            for cadena in ramas:
                actual = hoja
                for f in cadena:
                    actual = self._nuevo(f, actual)
                self._revisar_cierre(actual)
        nodo.aplicada = paso
        self.pasos.append({
            "paso": paso,
            "fórmula": (nodo.etiqueta + ": " if nodo.etiqueta else "") + a_texto(nodo.formula),
            "regla": regla,
            "tipo": "α (no ramifica)" if tipo == "alfa" else "β (ramifica)",
            "ramas afectadas": len(hojas),
        })

    # --- resultados -------------------------------------------------------
    def hojas(self):
        """Hojas de izquierda a derecha (mismo orden que el dibujo)."""
        resultado = []

        def recorrer(n):
            if not n.hijos:
                resultado.append(n)
            for h in n.hijos:
                recorrer(h)

        recorrer(self.raiz)
        return resultado

    def ramas_abiertas(self):
        return [h for h in self.hojas() if not h.cerrada]

    def ramas_cerradas(self):
        return [h for h in self.hojas() if h.cerrada]

    @staticmethod
    def literales(hoja):
        valores = {}
        n = hoja
        while n is not None:
            f = n.formula
            if f[0] == "atomo":
                valores[f[1]] = True
            elif f[0] == "no" and f[1][0] == "atomo":
                valores[f[1][1]] = False
            n = n.padre
        return valores


# ----------------------------------------------------------------------------
# 3b. SALIDA EN TEXTO
# ----------------------------------------------------------------------------
def modelo_texto(letras, valores):
    return ", ".join(f"{l} = {'V' if valores[l] else 'F'}" if l in valores else f"{l} = libre"
                     for l in letras)


def frase_modelo(significado, letras, valores):
    partes = []
    for l in letras:
        if l in valores:
            t = significado[l]
            t = t[0].lower() + t[1:]
            partes.append(t if valores[l] else "no " + t)
    return "; ".join(partes) if partes else "sin restricciones"


def _etiqueta_texto(n):
    t = (n.etiqueta + ": " if n.etiqueta else "") + a_texto(n.formula)
    return t + (" ✓" if n.aplicada is not None else "")


def _emitir(nodo, p1, p2):
    salida, actual, primero = [], nodo, True
    while True:
        salida.append((p1 if primero else p2) + _etiqueta_texto(actual))
        primero = False
        if len(actual.hijos) == 1:
            actual = actual.hijos[0]
            continue
        break
    if not actual.hijos:
        if actual.cerrada:
            c = actual.contradiccion
            salida.append(p2 + f"✕ rama cerrada ({c} y ¬{c})")
        else:
            salida.append(p2 + "● rama abierta")
    else:
        for i, h in enumerate(actual.hijos):
            ultimo = i == len(actual.hijos) - 1
            salida += _emitir(h, p2 + ("└── " if ultimo else "├── "), p2 + ("    " if ultimo else "│   "))
    return salida


def informe_texto(atomos, tablero):
    letras = list(atomos.significado)
    lineas = ["=== ÁRBOL SEMÁNTICO ===", "(✓ fórmula aplicada, ✕ rama cerrada, ● rama abierta)", ""]
    lineas += _emitir(tablero.raiz, "", "")
    abiertas, cerradas = tablero.ramas_abiertas(), tablero.ramas_cerradas()
    lineas += ["", "=== RESULTADO ==="]
    if not abiertas:
        lineas.append(f"La base es INCONSISTENTE: las {len(cerradas)} rama(s) se cierran. "
                      "Ninguna asignación hace verdaderas todas las reglas.")
    else:
        lineas.append(f"La base es CONSISTENTE: {len(abiertas)} rama(s) abierta(s) y {len(cerradas)} cerrada(s).")
        for i, hoja in enumerate(abiertas, 1):
            v = Tablero.literales(hoja)
            lineas += ["", f"Rama abierta {i}: {modelo_texto(letras, v)}",
                       f"Todas las reglas se cumplen cuando: {frase_modelo(atomos.significado, letras, v)}."]
    return "\n".join(lineas)


# ----------------------------------------------------------------------------
# 4. VISUALIZACIÓN
# ----------------------------------------------------------------------------
VERDE, ROJO, GRIS, AZUL = "#2e7d32", "#c62828", "#78909c", "#1565c0"


def figura_arbol(tablero, ancho_rama=190, alto_nivel=62):
    raiz = tablero.raiz
    pos = {}
    contador = [0]

    def colocar(nodo, nivel):
        if not nodo.hijos:
            x = contador[0]
            contador[0] += 1
        else:
            xs = [colocar(h, nivel + 1) for h in nodo.hijos]
            x = sum(xs) / len(xs)
        pos[nodo.idx] = (x, -nivel)
        return x

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
    colocar(raiz, 0)

    n_hojas = max(contador[0], 1)
    nivel_max = max(-y for _, y in pos.values())

    # aristas
    ex, ey = [], []
    for n in tablero.nodos:
        for h in n.hijos:
            ex += [pos[n.idx][0], pos[h.idx][0], None]
            ey += [pos[n.idx][1], pos[h.idx][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=GRIS, width=1.5),
                             hoverinfo="skip", showlegend=False))

    anotaciones = []
    for n in tablero.nodos:
        x, y = pos[n.idx]
        texto = (n.etiqueta + ": " if n.etiqueta else "") + a_texto(n.formula)
        if n.aplicada is not None:
            texto += f"  ✓{n.aplicada}"
        if n.hijos:
            fondo, borde = ("#e3f2fd" if n.etiqueta else "#f5f5f5"), (AZUL if n.etiqueta else GRIS)
        elif n.cerrada:
            fondo, borde = "#ffebee", ROJO
        else:
            fondo, borde = "#e8f5e9", VERDE
        anotaciones.append(dict(x=x, y=y, text=texto, showarrow=False, bgcolor=fondo,
                                bordercolor=borde, borderwidth=1.5, borderpad=4,
                                font=dict(size=12, color="#111111")))
        if not n.hijos:
            if n.cerrada:
                marca, color = f"✕ ({n.contradiccion} y ¬{n.contradiccion})", ROJO
            else:
                marca, color = "● abierta", VERDE
            anotaciones.append(dict(x=x, y=y - 0.62, text=f"<b>{marca}</b>", showarrow=False,
                                    font=dict(size=12, color=color)))

    # leyenda
    for nombre, color in (("fórmula / regla", AZUL), ("rama abierta", VERDE), ("rama cerrada ✕", ROJO)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nombre,
                                 marker=dict(size=11, color=color, symbol="square")))

    fig.update_layout(
        annotations=anotaciones,
        width=max(900, int(n_hojas * ancho_rama + 120)),
        height=int((nivel_max + 2.4) * alto_nivel),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.7, n_hojas - 0.3]),
        yaxis=dict(visible=False, range=[-(nivel_max + 1.5), 0.6]),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.0, x=0),
    )
    return fig


def mostrar_figura(fig):
    """Compatible con versiones antiguas y nuevas de Streamlit."""
    try:
        st.plotly_chart(fig, width="content")
    except TypeError:
        st.plotly_chart(fig, use_container_width=False)


# ----------------------------------------------------------------------------
# 5. INTERFAZ TKINTER
# ----------------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class Aplicacion:

    def __init__(self, root):

        self.root = root
        self.root.title("Tablero Semántico")
        self.root.geometry("1200x800")

        frame_superior = ttk.Frame(root)
        frame_superior.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            frame_superior,
            text="Base de conocimiento (una regla por línea)"
        ).pack(anchor="w")

        self.texto_entrada = scrolledtext.ScrolledText(
            frame_superior,
            height=10
        )
        self.texto_entrada.pack(fill="x")

        ejemplo = (
            "Si es de día entonces voy a trabajar\n"
            "Si llueve entonces no voy a trabajar\n"
            "Es de día"
        )

        self.texto_entrada.insert("1.0", ejemplo)

        ttk.Button(
            frame_superior,
            text="Generar tablero",
            command=self.generar
        ).pack(pady=5)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.tab_atomos = ttk.Frame(notebook)
        self.tab_formulas = ttk.Frame(notebook)
        self.tab_resultado = ttk.Frame(notebook)

        notebook.add(self.tab_atomos, text="Átomos")
        notebook.add(self.tab_formulas, text="Fórmulas")
        notebook.add(self.tab_resultado, text="Resultado")

        self.atomos_text = scrolledtext.ScrolledText(
            self.tab_atomos
        )
        self.atomos_text.pack(fill="both", expand=True)

        self.formulas_text = scrolledtext.ScrolledText(
            self.tab_formulas
        )
        self.formulas_text.pack(fill="both", expand=True)

        self.resultado_text = scrolledtext.ScrolledText(
            self.tab_resultado
        )
        self.resultado_text.pack(fill="both", expand=True)

    def generar(self):

        texto = self.texto_entrada.get("1.0", "end")

        try:

            reglas = separar_reglas(texto)

            atomos = TablaAtomos()

            formulas = [
                analizar(r, atomos)
                for r in reglas
            ]

            tablero = Tablero(formulas)
            
            fig = figura_arbol(tablero)
            fig.show()


        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e)
            )
            return

        self.atomos_text.delete("1.0", "end")
        self.formulas_text.delete("1.0", "end")
        self.resultado_text.delete("1.0", "end")

        self.atomos_text.insert(
            "end",
            "\n".join(
                f"{k} → {v}"
                for k, v in atomos.significado.items()
            )
        )

        salida_formulas = []

        for i, f in enumerate(formulas, 1):

            salida_formulas.append(
                f"R{i}: {a_texto(f)}"
            )

        self.formulas_text.insert(
            "end",
            "\n".join(salida_formulas)
        )

        self.resultado_text.insert(
            "end",
            informe_texto(atomos, tablero)
        )


def main():

    root = tk.Tk()

    Aplicacion(root)

    root.mainloop()


if __name__ == "__main__":
    main()