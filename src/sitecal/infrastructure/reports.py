import pandas as pd
from sitecal.core.calibration_engine import Calibration
import datetime
import pytz

def generate_markdown_report(
    calibration: Calibration,
    output_path: str,
    method: str
):
    """
    Generates a Markdown report with calibration results.
    """
    
    report_lines = []
    
    now = datetime.datetime.now(pytz.timezone('America/Santiago')).strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines.append("# Site Calibration Report")
    report_lines.append("")
    report_lines.append(f"Report generated on: {now}")
    report_lines.append("")
    report_lines.append(f"## Calibration Method: {method.upper()}")
    report_lines.append("")

    # Horizontal Transformation Parameters (2D)
    report_lines.append("### 🏗️ Ajuste Horizontal (2D)")
    report_lines.append("")
    if calibration.horizontal_params:
        hp = calibration.horizontal_params
        report_lines.append("- **Factor de Escala (a):** " + f"`{hp['a']:.6f}`")
        report_lines.append("- **Término de Rotación (b):** " + f"`{hp['b']:.6f}`")
        report_lines.append("- **Traslación Este:** " + f"`{hp['tE']:.3f} m`")
        report_lines.append("- **Traslación Norte:** " + f"`{hp['tN']:.3f} m`")
        
        # Derived: Scale and Rotation
        scale = (hp['a']**2 + hp['b']**2)**0.5
        import math
        rotation_rad = math.atan2(hp['b'], hp['a'])
        rotation_deg = math.degrees(rotation_rad)
        rotation_dms_d = int(rotation_deg)
        rotation_dms_m = int((abs(rotation_deg) - abs(rotation_dms_d)) * 60)
        rotation_dms_s = (abs(rotation_deg) - abs(rotation_dms_d) - rotation_dms_m/60) * 3600
        
        report_lines.append(f"- **Escala Implícita:** `{scale:.8f}`")
        report_lines.append(f"- **Rotación Implícita:** `{rotation_dms_d}° {rotation_dms_m}' {rotation_dms_s:.1f}\"`")

    else:
        report_lines.append("No se calcularon parámetros horizontales.")
    report_lines.append("")
    
    # Vertical Adjustment Parameters (Inclined Plane)
    report_lines.append("### 📐 Ajuste Vertical (1D)")
    report_lines.append("")
    if calibration.vertical_params:
        vp = calibration.vertical_params
        report_lines.append(f"- **Desplazamiento Vertical (Shift):** `{vp['vertical_shift']:.3f} m`")
        report_lines.append(f"- **Inclinación Norte:** `{vp['slope_north']*1e6:.2f} ppm`")
        report_lines.append(f"- **Inclinación Este:** `{vp['slope_east']*1e6:.2f} ppm`")
        report_lines.append(f"- **Centroide (N, E):** `({vp['centroid_north']:.3f}, {vp['centroid_east']:.3f})`")
    else:
        report_lines.append("No se calcularon parámetros verticales.")
    report_lines.append("")

    # Residuals Table
    report_lines.append("### Residuals (mm)")
    report_lines.append("")
    if calibration.residuals is not None:
        residuals_mm = calibration.residuals.copy()
        residuals_mm["dE"] = (residuals_mm["dE"] * 1000).round(1)
        residuals_mm["dN"] = (residuals_mm["dN"] * 1000).round(1)
        residuals_mm["dH"] = (residuals_mm["dH"] * 1000).round(1)
        residuals_mm.rename(columns={"dE": "dE (mm)", "dN": "dN (mm)", "dH": "dH (mm)"}, inplace=True)
        report_lines.append(residuals_mm.to_markdown(index=False))
    else:
        report_lines.append("No residuals were calculated.")
    report_lines.append("")

    # Statistics
    report_lines.append("### Statistics")
    report_lines.append("")
    if calibration.residuals is not None:
        residuals = calibration.residuals
        # Calculate horizontal error
        residuals["error_h"] = (residuals["dE"]**2 + residuals["dN"]**2)**0.5
        
        worst_point = residuals.loc[residuals["error_h"].idxmax()]
        best_point = residuals.loc[residuals["error_h"].idxmin()]
        std_dev = residuals[["dE", "dN", "dH"]].std().to_dict()
        percentile_99 = residuals["error_h"].quantile(0.99)

        report_lines.append(f"- **Worst Point:** `{worst_point['Point']}` (Error: {(worst_point['error_h'] * 1000):.1f} mm)")
        report_lines.append(f"- **Best Point:** `{best_point['Point']}` (Error: {(best_point['error_h'] * 1000):.1f} mm)")
        report_lines.append("- **Standard Deviations (mm):**")
        for axis, value in std_dev.items():
            report_lines.append(f"  - `{axis}`: {(value * 1000):.1f} mm")
        report_lines.append(f"- **99th Percentile of Horizontal Errors:** {(percentile_99 * 1000):.1f} mm")

    else:
        report_lines.append("Statistics could not be calculated.")
    report_lines.append("")

    return "\n".join(report_lines) + "\n"
