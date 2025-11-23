def classify_risk(mean_width_mm: float) -> str:
    """
    Clasifica el nivel de riesgo de una fisura según su ancho medio (en mm).

    Criterio basado en literatura de ingeniería civil:
    - Bajo:  ancho <= 0.3 mm
    - Medio: 0.3 < ancho < 3.0 mm
    - Alto:  ancho >= 3.0 mm

    El sistema también maneja casos degenerados (ancho negativo o vacío).
    """

    if mean_width_mm is None or mean_width_mm <= 0:
        return "Sin datos"

    if mean_width_mm <= 0.3:
        return "Bajo"
    elif mean_width_mm < 3.0:
        return "Medio"
    else:
        return "Alto"
