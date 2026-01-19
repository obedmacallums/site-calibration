from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Calibration(ABC):
    @abstractmethod
    def train(self, df_local: pd.DataFrame, df_global: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class Similarity2D(Calibration):
    def __init__(self):
        self.horizontal_params = None
        self.vertical_params = None
        self.residuals = None

    def train(self, df_local: pd.DataFrame, df_global: pd.DataFrame):
        """
        Calculates 2D similarity transformation parameters (a, b, tE, tN)
        and Vertical Adjustment parameters (Inclined Plane or Constant Shift).
        """
        
        merged_df = pd.merge(df_local, df_global, on="Point", suffixes=('_local', '_global'))
        n = len(merged_df)

        # --- Horizontal (2D Similarity) ---
        x = merged_df["Easting_global"].values
        y = merged_df["Northing_global"].values
        E = merged_df["Easting_local"].values
        N = merged_df["Northing_local"].values
        
        # Calculate centroids
        x_c = np.mean(x)
        y_c = np.mean(y)
        E_c = np.mean(E)
        N_c = np.mean(N)
        
        # Center coordinates
        x_prime = x - x_c
        y_prime = y - y_c
        E_prime = E - E_c
        N_prime = N - N_c

        # Solve for a and b using centered coordinates
        A = np.zeros((2 * n, 2))
        A[:n, 0] = x_prime
        A[:n, 1] = -y_prime
        A[n:, 0] = y_prime
        A[n:, 1] = x_prime

        L = np.concatenate([E_prime, N_prime])

        params_ab, _, _, _ = np.linalg.lstsq(A, L, rcond=None)
        a = params_ab[0]
        b = params_ab[1]
        
        # Calculate translations
        tE = E_c - a * x_c + b * y_c
        tN = N_c - b * x_c - a * y_c

        self.horizontal_params = {"a": a, "b": b, "tE": tE, "tN": tN}

        # --- Vertical (Inclined Plane) ---
        # Z_error = Z_global - Z_local
        # Model: Z_error = C + S_N * (N_local - N_c) + S_E * (E_local - E_c)
        
        # Get heights. Handle varying column names.
        h_global = merged_df["h_global"].values if "h_global" in merged_df.columns else np.zeros(n)
        h_local = merged_df["h_local"].values if "h_local" in merged_df.columns else np.zeros(n)
        
        Z_error = h_global - h_local
        
        if n >= 3:
            # Planar fit
            # A matrix: [1, (N - N_c), (E - E_c)]
            # We use local coordinates for the domain of the inclination as per standard practice (or translated global)
            # TBC usually applies inclination based on position. Let's use Local Centering.
            
            A_v = np.ones((n, 3))
            A_v[:, 1] = N_prime # Using N_prime (N_local - N_c) is a good approximation for centering
            A_v[:, 2] = E_prime
            
            # Solve for [C, S_N, S_E]
            v_params, _, _, _ = np.linalg.lstsq(A_v, Z_error, rcond=None)
            C = v_params[0]
            slope_n = v_params[1]
            slope_e = v_params[2]
        else:
            # Constant shift only
            C = np.mean(Z_error)
            slope_n = 0.0
            slope_e = 0.0
            
        self.vertical_params = {
            "vertical_shift": C,
            "slope_north": slope_n,
            "slope_east": slope_e,
            "centroid_north": N_c,
            "centroid_east": E_c
        }
        
        # Calculate residuals & Transformed Values
        transformed = self.transform(merged_df)
        self.residuals = pd.DataFrame({
            "Point": merged_df["Point"],
            "dE": transformed["Easting"] - merged_df["Easting_local"],
            "dN": transformed["Northing"] - merged_df["Northing_local"],
            "dH": transformed["h"] - merged_df["h_local"] - (merged_df["h_global"] - merged_df["h_local"]) 
            # Note on dH residual: 
            # Transformed h = h_local + Correction. 
            # Residual = (Transformed h) - h_global 
            # Wait, dH is usually (Transformed - Measured/Control). 
            # Here Global is Control.
            # dH = Transformed["h"] - h_global
        })
        # let's correct dH calc based on TBC definition: dH = Source + Shift - Control
        self.residuals["dH"] = transformed["h"] - merged_df.get("h_global", 0)


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.horizontal_params is None or self.vertical_params is None:
            raise RuntimeError("The calibration model has not been trained.")
        
        # Horizontal
        a = self.horizontal_params["a"]
        b = self.horizontal_params["b"]
        tE = self.horizontal_params["tE"]
        tN = self.horizontal_params["tN"]

        # Vertical
        C = self.vertical_params["vertical_shift"]
        Sn = self.vertical_params["slope_north"]
        Se = self.vertical_params["slope_east"]
        Nc = self.vertical_params["centroid_north"]
        Ec = self.vertical_params["centroid_east"]

        # Handle column names dynamically
        if "Easting_global" in df.columns:
            x = df["Easting_global"].values
            y = df["Northing_global"].values
        else:
            x = df["Easting"].values
            y = df["Northing"].values
            
        if "h_local" in df.columns:
            h_local = df["h_local"].values
        elif "h" in df.columns:
             h_local = df["h"].values
        else:
             h_local = np.zeros(len(df))

        # Apply 2D Sim
        E_trans = a * x - b * y + tE
        N_trans = b * x + a * y + tN
        
        # Apply Vertical Adjustment
        # We need N_local and E_local for the plane. 
        # If we only have global input (transforming global to local), we use the transformed values as proxy for position
        # Or if we have local columns in input.
        
        # The plane Z_err = C + Sn*(N - Nc) + Se*(E - Ec)
        # Z_local_derived = Z_global - Z_err (if subtracting error) or Z_local + Z_adj = Z_global
        # Let's align with the training: Z_error = Z_global - Z_local
        # So Z_global = Z_local + Z_error
        # We want to output "Transformed" coords. Usually this means transforming GNSS (Global) -> Local Grid.
        # But wait, TBC Site Cal transforms GPS (Global) to Grid (Local).
        # My train code defined a,b,tE,tN such that Local = f(Global).
        # So E_trans IS the estimated Local Easting.
        # So we can use E_trans and N_trans for the inclined plane domain.
        
        dZ = C + Sn * (N_trans - Nc) + Se * (E_trans - Ec)
        
        # wait, input 'h' in transform might be global h?
        # If input is global list, we want to match Local.
        # In train: Z_error = H_global - H_local.
        # This implies H_global = H_local + Z_error => H_local = H_global - Z_error.
        # If the user passes global coords to transform(), they expect Local Grid coords out.
        # So H_out = H_in (Global) - dZ.
        
        # Let's verify standard: Site Cal applied to GPS point:
        # 1. Convert Lat/Lon to Transverse Mercator (if not already) -> Global Grid
        # 2. Apply Horizontal Adjust -> Local Grid N,E
        # 3. Apply Vertical Adjust -> Local Elev.
        
        # If Z_error was defined as Global - Local (positive means Global is higher)
        # Then Local = Global - Z_error. 
        # Yes.
        
        H_trans = h_local - dZ # Wait, h_local variable name in this func might be misleading if it comes from 'h' column of input which might be global.
        # Let's assume input to transform is "Source" (Global).
        
        return pd.DataFrame({
            "Point": df["Point"],
            "Easting": E_trans,
            "Northing": N_trans,
            "h": H_trans
        })



class CalibrationFactory:
    @staticmethod
    def create(method: str) -> Calibration:
        if method == "tbc" or method == "ltm":
            return Similarity2D()
        else:
            raise ValueError(f"Unknown calibration method: {method}")