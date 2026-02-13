#!/usr/bin/env python3
"""
Material models for plasticity computation.
This module contains different material constitutive models for plasticity analysis.
"""

import numpy as np
import ast
from pathlib import Path
from scipy.optimize import newton, root

# --- Utility Functions ---

def parse_material_params(file_path):
    """
    Parses a single material parameter file with 'key = value' format.
    Cross-platform compatible with explicit UTF-8 encoding.
    """
    params = {}
    # Convert to Path object for cross-platform compatibility
    file_path = Path(file_path)
    
    # Use UTF-8 encoding explicitly for cross-platform compatibility
    with open(file_path, 'r', encoding='utf-8', newline=None) as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                # Use ast.literal_eval for safe evaluation of Python literals
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # Fallback for simple strings if needed
                params[key] = value
    return params

def load_and_create_model(kinematic_file, isotropic_file=None, model_type='uniaxial'):
    """
    Enhanced version: Loads parameters from files and creates either uniaxial or multi-axial material models.
    
    Args:
        kinematic_file: Path to kinematic hardening parameters file
        isotropic_file: Path to isotropic hardening parameters file (optional)
        model_type: 'uniaxial' or 'multiaxial' to specify which model to create
    
    Returns:
        For uniaxial: (model, E, yield_stress)
        For multiaxial: (model, E, yield_stress, nu)
    """
    # Load kinematic parameters
    kin_params = parse_material_params(kinematic_file)
    
    # Load isotropic parameters if a file is provided
    iso_params = {}
    if isotropic_file:
        iso_params = parse_material_params(isotropic_file)
        
    # Merge parameters from both files
    all_params = {**kin_params, **iso_params}
    
    # Normalize parameter names from file keys to class __init__ arguments
    final_params = {}
    param_map = {
        'c': 'C',
        'y': 'gamma',
        'sigy': 'yield_stress',
        'Q': 'R_inf'
        # 'b', 'm', 'E', 'nu' are kept as is
    }
    for key, value in all_params.items():
        final_params[param_map.get(key, key)] = value

    # Extract common parameters required by the solver
    E = final_params.pop('E')
    yield_stress = final_params.pop('yield_stress')
    nu = final_params.pop('nu', 0.3)  # Default Poisson's ratio for multiaxial
    
    # Create isotropic hardening model
    if 'R_inf' in final_params or 'b' in final_params:
        R_inf = final_params.pop('R_inf', 0)
        b = final_params.pop('b', 0)
        isotropic_model = VoceIsotropicHardeningModel(R_inf=R_inf, b=b)
    else:
        isotropic_model = NoIsotropicHardeningModel()

    # Decide which material model to create based on model_type and kinematic file name
    # Use Path for cross-platform compatibility
    model_name = Path(kinematic_file).name
    
    if model_type == 'uniaxial':
        if 'ChabocheT' in model_name or 'NLK-T' in model_name or 'NLKT' in model_name:
            # ChabocheT model requires 'al' parameter (threshold)
            if 'al' not in final_params:
                raise ValueError(f"ChabocheT model requires 'al' parameter (threshold), but not found. Please specify 'al' in parameter file.")
            model = ChabocheTModelUniAxial(isotropic_model=isotropic_model, **final_params)
        elif 'Chaboche' in model_name:
            model = ChabocheModelUniAxial(isotropic_model=isotropic_model, **final_params)
        elif 'OWII' in model_name or 'OhnoWang' in model_name:
            model = OhnoWangIIModelUniAxial(isotropic_model=isotropic_model, **final_params)
        elif 'AKO' in model_name or 'AbdelKarim' in model_name or 'Abdel-Karim' in model_name:
            # AKO model requires mu parameter (instead of m)
            if 'mu' not in final_params and 'm' in final_params:
                # If 'm' is provided, use it as mu (with warning) or use default
                # For now, raise error to force explicit mu parameter
                raise ValueError(f"AKO model requires 'mu' parameter, but only 'm' found. Please specify 'mu' in parameter file.")
            model = AbdelKarimOhnoModelUniAxial(isotropic_model=isotropic_model, **final_params)
        elif 'Kang' in model_name:
            # Kang model requires mu parameter (similar to AKO)
            if 'mu' not in final_params and 'm' in final_params:
                raise ValueError(f"Kang model requires 'mu' parameter, but only 'm' found. Please specify 'mu' in parameter file.")
            model = KangModelUniAxial(isotropic_model=isotropic_model, **final_params)
        else:
            raise ValueError(f"Could not determine uniaxial model type from filename: {model_name}")
        return model, E, yield_stress
        
    elif model_type == 'multiaxial':
        if 'Chaboche' in model_name:
            model = ChabocheModelMultiAxial(isotropic_model=isotropic_model, E=E, nu=nu, **final_params)
        elif 'OWII' in model_name or 'OhnoWang' in model_name:
            model = OhnoWangIIModelMultiAxial(isotropic_model=isotropic_model, E=E, nu=nu, **final_params)
        else:
            raise ValueError(f"Could not determine multiaxial model type from filename: {model_name}")
        return model, E, yield_stress, nu
    
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'uniaxial' or 'multiaxial'")

# Keep the original function for backward compatibility
def load_and_create_model_legacy(kinematic_file, isotropic_file=None):
    """
    Legacy version: Loads parameters from files, creates the appropriate uniaxial material model,
    and returns the model along with E and yield_stress for the solver.
    """
    return load_and_create_model(kinematic_file, isotropic_file, 'uniaxial')

def load_and_create_ako_model(kinematic_file, isotropic_file=None, model_type='multiaxial'):
    """
    Loads parameters from files and creates a multi-axial Abdel-Karim & Ohno material model.
    Args:
        kinematic_file: Path to kinematic hardening parameters file
        isotropic_file: Path to isotropic hardening parameters file (optional)
        model_type: only 'multiaxial' supported for AK-O
    Returns:
        model, E, yield_stress, nu
    """
    kin_params = parse_material_params(kinematic_file)
    iso_params = {}
    if isotropic_file:
        iso_params = parse_material_params(isotropic_file)
    all_params = {**kin_params, **iso_params}
    final_params = {}
    param_map = {
        'c': 'C',
        'y': 'gamma',
        'mu': 'mu',
        'sigy': 'yield_stress',
        'Q': 'R_inf'
    }
    for key, value in all_params.items():
        final_params[param_map.get(key, key)] = value
    E = final_params.pop('E')
    yield_stress = final_params.pop('yield_stress')
    nu = final_params.pop('nu', 0.3)
    if 'R_inf' in final_params or 'b' in final_params:
        R_inf = final_params.pop('R_inf', 0)
        b = final_params.pop('b', 0)
        isotropic_model = VoceIsotropicHardeningModel(R_inf=R_inf, b=b)
    else:
        isotropic_model = NoIsotropicHardeningModel()
    model = AbdelKarimOhnoModelMultiAxial(isotropic_model=isotropic_model, E=E, nu=nu, **final_params)
    return model, E, yield_stress, nu

def load_and_create_kang_model(kinematic_file, isotropic_file=None, model_type='multiaxial'):
    """
    Loads parameters from files and creates a multi-axial Kang material model.
    Accepts both lowercase and uppercase keys for all parameters (e.g., 'e' and 'E', 'nu' and 'NU', etc.).
    Args:
        kinematic_file: Path to kinematic hardening parameters file
        isotropic_file: Path to isotropic hardening parameters file (optional)
        model_type: only 'multiaxial' supported for Kang
    Returns:
        model, E, yield_stress, nu
    """
    kin_params = parse_material_params(kinematic_file)
    iso_params = {}
    if isotropic_file:
        iso_params = parse_material_params(isotropic_file)
    all_params = {**kin_params, **iso_params}
    final_params = {}
    # Accept both lowercase and uppercase for all keys, and map parameter names
    param_map = {
        'c': ['c', 'C'],
        'gamma': ['gamma', 'y', 'Y'],
        'mu': ['mu', 'MU'],
        'e': ['e', 'E'],
        'nu': ['nu', 'NU'],
        'sigy': ['sigy', 'SIGY'],
        'R_inf': ['Q', 'R_inf'],  # Map Q to R_inf for isotropic hardening
        'b': ['b', 'B'],
    }
    for k, aliases in param_map.items():
        found = False
        for alias in aliases:
            if alias in all_params:
                final_params[k] = all_params[alias]
                found = True
                break
        if not found and k not in ['R_inf', 'b']:  # R_inf and b are optional
            raise KeyError(f"Parameter '{k}' (aliases: {aliases}) not found in material file.")
    
    # Extract common parameters
    E = final_params.pop('e')
    yield_stress = final_params.pop('sigy')
    nu = final_params.pop('nu', 0.3)
    
    # Create isotropic hardening model (same logic as other models)
    if 'R_inf' in final_params or 'b' in final_params:
        R_inf = final_params.pop('R_inf', 0)
        b = final_params.pop('b', 0)
        isotropic_model = VoceIsotropicHardeningModel(R_inf=R_inf, b=b)
    else:
        isotropic_model = NoIsotropicHardeningModel()
    
    # Create Kang model with remaining parameters
    model = KangModelMultiAxial(isotropic_model, final_params['c'], final_params['gamma'], final_params['mu'], E, nu)
    return model, E, yield_stress, nu

# --- Multi-axial Tensor Utilities ---

def deviatoric(tensor):
    """Computes the deviatoric part of a 3x3 tensor."""
    return tensor - (1./3.) * np.trace(tensor) * np.identity(3)

def mag(tensor):
    """Computes the Frobenius norm of a tensor, equivalent to sqrt(T:T)."""
    return np.sqrt(np.sum(tensor * tensor))

def tensor_to_voigt(tensor):
    """Converts a 3x3 symmetric tensor to a 6x1 Voigt notation vector."""
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[1, 2], tensor[0, 2]])

def voigt_to_tensor(voigt):
    """Converts a 6x1 Voigt notation vector to a 3x3 symmetric tensor."""
    tensor = np.zeros((3, 3))
    tensor[0, 0], tensor[1, 1], tensor[2, 2] = voigt[0], voigt[1], voigt[2]
    tensor[0, 1] = tensor[1, 0] = voigt[3]
    tensor[1, 2] = tensor[2, 1] = voigt[4]
    tensor[0, 2] = tensor[2, 0] = voigt[5]
    return tensor

# --- Isotropic Hardening Models ---

class BaseIsotropicHardeningModel:
    """
    Abstract base class for isotropic hardening models.
    """
    def __init__(self, **params):
        self.params = params
        
    def compute_hardening_modulus(self, R, p):
        """
        Compute the isotropic hardening modulus h_iso.
        
        Args:
            R: Current isotropic hardening variable
            p: Accumulated plastic strain
            
        Returns:
            h_iso: Isotropic hardening modulus
        """
        raise NotImplementedError
        
    def update_explicit(self, R, p, dp):
        """
        Update isotropic hardening using explicit method.
        
        Args:
            R: Current isotropic hardening variable
            p: Current accumulated plastic strain
            dp: Plastic strain increment
            
        Returns:
            tuple: (R_new, p_new)
        """
        raise NotImplementedError
        
    # def update_implicit(self, R, p, dp):
    #     """
    #     Update isotropic hardening using implicit method.
        
    #     Args:
    #         R: Current isotropic hardening variable
    #         p: Current accumulated plastic strain
    #         dp: Plastic strain increment
            
    #     Returns:
    #         tuple: (R_new, p_new)
    #     """
    #     raise NotImplementedError


class VoceIsotropicHardeningModel(BaseIsotropicHardeningModel):
    """
    Voce isotropic hardening model implementation.
    Follows the evolution law: dR = b * (R_inf - R) * dp
    """
    def __init__(self, R_inf=0, b=0):
        super().__init__(R_inf=R_inf, b=b)
        self.R_inf = R_inf  # Saturation value for isotropic hardening
        self.b = b          # Rate parameter for isotropic hardening
        
    def compute_hardening_modulus(self, R, p):
        """
        Compute the Voce isotropic hardening modulus.
        
        Args:
            R: Current isotropic hardening variable
            p: Accumulated plastic strain (not used in Voce model)
            
        Returns:
            h_iso: Isotropic hardening modulus
        """
        if self.b == 0:
            return 0.0
        return self.b * (self.R_inf - R)
        
    def update_explicit(self, R, p, dp):
        """
        Update isotropic hardening using explicit Voce method.
        
        Args:
            R: Current isotropic hardening variable
            p: Current accumulated plastic strain
            dp: Plastic strain increment
            
        Returns:
            tuple: (R_new, p_new)
        """
        # Update accumulated plastic strain
        p_new = p + dp
        
        # Update isotropic hardening variable R using explicit method
        if self.b == 0:
            R_new = R
        else:
            dR = self.b * (self.R_inf - R) * dp
            R_new = R + dR
            
            # Apply saturation bounds
            if self.R_inf > 0 and R_new > self.R_inf:
                R_new = self.R_inf
            elif self.R_inf < 0 and R_new < self.R_inf:
                R_new = self.R_inf
                
        return R_new, p_new
        
        
    def update_implicit(self, R, p, dp):
        """
        Update isotropic hardening using implicit Voce method WITHOUT saturation bounds.
        This matches the legacy multi-axial implementation exactly.
        
        Args:
            R: Current isotropic hardening variable
            p: Current accumulated plastic strain
            dp: Plastic strain increment
            
        Returns:
            tuple: (R_new, p_new)
        """
        # Update accumulated plastic strain
        p_new = p + dp
        
        # Update isotropic hardening variable R using implicit method
        if self.b == 0:
            R_new = R
        else:
            # Implicit update: R_new = (R + b*R_inf*dp) / (1 + b*dp)
            R_new = (R + self.b * self.R_inf * dp) / (1 + self.b * dp)
            # NO saturation bounds applied - this matches legacy behavior
                
        return R_new, p_new


class AdvancedVoceIsotropicHardeningModel(BaseIsotropicHardeningModel):
    """
    Advanced Voce isotropic hardening with a memory-dependent saturation Q(q).
    - Q(q) = Q_M + (Q_0 - Q_M) * exp(-2 * mu * q)
    - R evolution keeps Voce structure: R_dot = b * (Q(q) - R) * p_dot
    Notes:
      - This class exposes set_q(q) so that upper layers can feed the observed
        plastic half-range at half-cycle completion. Default q starts from 0.0.
      - The explicit/implicit updates mirror Voce, with R_inf replaced by Q(q).
    """
    def __init__(self, Q0=0.0, QM=0.0, mu=0.0, b=0.0, q0=0.0):
        super().__init__(Q0=Q0, QM=QM, mu=mu, b=b, q0=q0)
        self.Q0 = Q0
        self.QM = QM
        self.mu = mu
        self.b = b
        self.q = q0

    # --- Public API for memory variable ---
    def set_q(self, q_value):
        # Monotonic memory (only increases) to reflect memorization surface
        if q_value is None:
            return
        if q_value > self.q:
            self.q = float(q_value)

    # --- Helpers ---
    def _Q_of_q(self):
        return self.QM + (self.Q0 - self.QM) * np.exp(-2.0 * self.mu * self.q)

    # --- Interface methods ---
    def compute_hardening_modulus(self, R, p):
        # h_iso = dR/dp with p being accumulated plastic strain
        return self.b * (self._Q_of_q() - R)

    def update_explicit(self, R, p, dp):
        p_new = p + dp
        Qq = self._Q_of_q()
        R_new = R + self.b * (Qq - R) * dp
        return R_new, p_new

    def update_implicit(self, R, p, dp):
        # Backward Euler using Q(q) at the beginning of increment to avoid extra coupling
        p_new = p + dp
        Qq = self._Q_of_q()
        if self.b == 0:
            return R, p_new
        R_new = (R + self.b * Qq * dp) / (1.0 + self.b * dp)
        return R_new, p_new

class NoIsotropicHardeningModel(BaseIsotropicHardeningModel):
    """
    No isotropic hardening model (purely kinematic hardening).
    """
    def __init__(self):
        super().__init__()
        
    def compute_hardening_modulus(self, R, p):
        """No isotropic hardening contribution."""
        return 0.0
        
    def update_explicit(self, R, p, dp):
        """No isotropic hardening update."""
        return R, p + dp
        
    # def update_implicit(self, R, p, dp):
    #     """No isotropic hardening update."""
    #     return R, p + dp
        
    def update_implicit(self, R, p, dp):
        # No isotropic hardening; still accumulate equivalent plastic strain p
        return R, p + dp 

# --- Base Material Model ---

class BaseMaterialModel:
    """
    Abstract base class for material constitutive models.
    """
    def __init__(self, **params):
        self.params = params

    def compute_trial_stress(self, total_strain, plastic_strain, E):
        """
        Compute the trial stress for a given total strain and plastic strain.
        """
        return E * (total_strain - plastic_strain)

    def yield_function(self, trial_stress, total_backstress, sigma_y):
        """
        Evaluate the yield function.
        """
        shifted_stress = trial_stress - total_backstress
        return np.abs(shifted_stress) - sigma_y, shifted_stress

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the plastic modulus (kinematic + isotropic hardening).
        This is a generic method that should be implemented by specific material models.
        """
        raise NotImplementedError

    def get_material_params(self, param_name):
        """
        Get material parameters by name.
        This method provides controlled access to material properties.
        """
        if hasattr(self, param_name):
            return getattr(self, param_name)
        else:
            raise AttributeError(f"Material model does not have parameter '{param_name}'")

    def plastic_multiplier(self, f, E, backstress, state_vars):
        """
        Compute the plastic multiplier (d_epsilon).
        """
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = E + h_total
        
        # Prevent division by zero or negative values in explicit integration
        d_epsilon = f / denominator if denominator > 1e-12 else 0.0
        if d_epsilon < 0:
            d_epsilon = 0.0
            
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, sign, state_vars):
        """
        Update the backstress components.
        """
        raise NotImplementedError

    def update_state_vars(self, state_vars, d_epsilon, sign):
        """
        Update any additional state variables (if needed).
        """
        return state_vars

# --- Material Models ---

class ChabocheModelUniAxial(BaseMaterialModel):
    """
    Chaboche kinematic hardening model implementation.
    """
    def __init__(self, isotropic_model, C, gamma):
        super().__init__(C=C, gamma=gamma)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.n_components = len(C)
        # Isotropic hardening model
        self.isotropic_model = isotropic_model

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Chaboche plastic modulus components.
        Returns total hardening modulus (h_kin + h_iso).
        """
        sign = state_vars.get('sign', 1.0)
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # Kinematic hardening modulus
        h_kin = np.sum(self.C - self.gamma * sign * backstress)

        # Isotropic hardening modulus
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)

        return h_kin + h_iso

    def plastic_multiplier(self, f, E, backstress, state_vars):
        # Use the generic plastic modulus computation
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = E + h_total
        
        # Prevent division by zero or negative values in explicit integration
        d_epsilon = f / denominator if denominator > 1e-12 else 0.0
        if d_epsilon < 0:
            d_epsilon = 0.0
            
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, sign, state_vars):
        new_backstress = backstress.copy()
        for i in range(self.n_components):
            delta_alpha = self.C[i] * d_epsilon * sign - self.gamma[i] * backstress[i] * d_epsilon
            new_backstress[i] += delta_alpha
        return new_backstress

    def update_backstress_implicit(self, backstress, d_epsilon, sign):
        """
        Update backstress using implicit method (Backward Euler).
        Solves: alpha_new = alpha_old + C*dp*sign - gamma * alpha_new * dp
        => alpha_new * (1 + gamma * dp) = alpha_old + C * dp * sign
        => alpha_new = (alpha_old + C * dp * sign) / (1 + gamma * dp)
        """
        new_backstress = backstress.copy()
        
        # Vectorized implementation since C and gamma are arrays
        # But backstress is a list or array? In UniAxial it is an array of components.
        # self.C and self.gamma are arrays.
        
        # d_epsilon is dp (scalar)
        dp = d_epsilon
        
        for i in range(self.n_components):
            alpha_old = backstress[i]
            C = self.C[i]
            gam = self.gamma[i]
            
            numerator = alpha_old + C * dp * sign
            denominator = 1.0 + gam * dp
            
            new_backstress[i] = numerator / denominator
            
        return new_backstress

    def update_state_vars(self, state_vars, d_epsilon, sign):
        """
        Update isotropic hardening state variables (R and p) using separate isotropic hardening model.
        """
        # Get current values from the solver state
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # d_epsilon is the increment of accumulated plastic strain (dp)
        dp = d_epsilon
        
        # Update isotropic hardening using separate model (explicit method)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        
        # Update state variables
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        
        return state_vars
class ChabocheTModelUniAxial(BaseMaterialModel):
    """
    1D Chaboche with threshold on the 4th backstress component (NLK-T).
    Evolution (1D explicit):
      a4_dot = C4 * dp * sign - gamma4 * (1 - al / J(a4))_+ * a4 * dp
    where J(a4) is the J2-equivalent of backstress. In 1D, we align with
    the multi-axial convention by using J(a4) = sqrt(3/2) * |a4| to be
    consistent with von Mises mapping between 3D and 1D.
    Other components i=1..3 follow standard 1D Chaboche.
    """
    def __init__(self, isotropic_model, C, gamma, al):
        super().__init__(C=C, gamma=gamma, al=al)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.n_components = len(C)
        assert self.n_components >= 4, "ChabocheT requires at least 4 components (last is thresholded)."
        self.al = al  # threshold X_l
        self.isotropic_model = isotropic_model

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the ChabocheT plastic modulus.
        Components 0-2: Standard Chaboche
        Component 3: Thresholded Chaboche with factor (1 - al/J(a4))_+
        where J(a4) = sqrt(3/2) * |a4| (consistent with multi-axial convention)
        """
        sign = state_vars.get('sign', 1.0)
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)

        # Standard Chaboche for components 0..n-2, modified for the 4th (index 3)
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            if i == 3:  # thresholded component
                # J(a4) = sqrt(3/2) * |a4| (consistent with multi-axial convention)
                ai_abs = abs(ai)
                Ja = np.sqrt(3.0/2.0) * ai_abs
                # Effective gamma reduced by (1 - al / J(a4))_+
                factor = max(0.0, 1.0 - (self.al / Ja) if Ja > 1e-16 else 0.0)
                h_kin += self.C[i] - self.gamma[i] * factor * sign * ai
            else:
                h_kin += self.C[i] - self.gamma[i] * sign * ai

        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        return h_kin + h_iso

    def plastic_multiplier(self, f, E, backstress, state_vars):
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = E + h_total
        d_epsilon = f / denominator if denominator > 1e-12 else 0.0
        if d_epsilon < 0:
            d_epsilon = 0.0
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, sign, state_vars):
        """
        Update backstress using explicit ChabocheT evolution law.
        Components 0-2: Standard Chaboche
        Component 3: Thresholded Chaboche with factor (1 - al/J(a4))_+
        """
        new_backstress = backstress.copy()
        for i in range(self.n_components):
            if i == 3:  # thresholded component
                ai = backstress[i]
                # J(a4) = sqrt(3/2) * |a4| (consistent with multi-axial convention)
                ai_abs = abs(ai)
                Ja = np.sqrt(3.0/2.0) * ai_abs
                # Factor: (1 - al / J(a4))_+
                factor = max(0.0, 1.0 - (self.al / Ja) if Ja > 1e-16 else 0.0)
                delta_alpha = self.C[i] * d_epsilon * sign - self.gamma[i] * factor * ai * d_epsilon
            else:
                # Standard Chaboche for components 0-2
                delta_alpha = self.C[i] * d_epsilon * sign - self.gamma[i] * backstress[i] * d_epsilon
            new_backstress[i] += delta_alpha
        return new_backstress

    def update_backstress_implicit(self, backstress, d_epsilon, sign):
        """
        Update backstress using implicit method (Backward Euler) for ChabocheT.
        
        Components 0-2: Standard Chaboche (analytical solution)
        Component 3: Thresholded Chaboche (non-linear, requires numerical solution)
        
        Implicit equation for component 3:
        α_new = α_old + C·dp·sign - γ·(1 - al/J(α_new))_+ · α_new·dp
        where J(α_new) = sqrt(3/2) * |α_new|
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            alpha_old = backstress[i]
            C = self.C[i]
            gam = self.gamma[i]
            
            if i == 3:  # thresholded component (non-linear)
                # RHS = alpha_old + C * dp * sign
                RHS = alpha_old + C * dp * sign
                RHS_mag = abs(RHS)
                RHS_sign = np.sign(RHS) if abs(RHS) > 1e-16 else (1.0 if sign > 0 else -1.0)
                
                # If no recovery (gamma = 0) or no step, solution is trivial
                if gam == 0 or dp < 1e-16:
                    new_backstress[i] = RHS
                    continue
                
                # Implicit equation: α_new + γ·(1 - al/J(α_new))_+ · α_new·dp = RHS
                # Solve for magnitude y = |α_new|
                # y * [1 + γ·dp·(1 - al/(sqrt(3/2)*y))_+] = |RHS|
                def residual_mag(y):
                    if y < 0:
                        return -abs(RHS_mag) - y
                    
                    # J(α_new) = sqrt(3/2) * y
                    Ja = np.sqrt(3.0/2.0) * y
                    
                    # Factor: (1 - al / J(α_new))_+
                    factor = max(0.0, 1.0 - (self.al / Ja) if Ja > 1e-16 else 0.0)
                    
                    # Recovery coefficient
                    recovery_coeff = gam * factor
                    
                    # Implicit equation: y * (1 + recovery_coeff * dp) = |RHS|
                    return y * (1.0 + recovery_coeff * dp) - RHS_mag
                
                # Bracket search: [0, |RHS|]
                lower = 0.0
                upper = RHS_mag
                
                # Check if upper bound gives positive residual
                if residual_mag(upper) <= 0:
                    upper = RHS_mag * 2.0
                    max_iter = 10
                    for _ in range(max_iter):
                        if residual_mag(upper) > 0:
                            break
                        upper *= 2.0
                
                # Solve using brentq
                try:
                    from scipy.optimize import brentq
                    y_sol = brentq(residual_mag, lower, upper, xtol=1e-12, maxiter=50)
                    x = y_sol * RHS_sign
                except Exception as e:
                    # Fallback: use explicit update
                    ai_abs = abs(alpha_old)
                    Ja = np.sqrt(3.0/2.0) * ai_abs
                    factor = max(0.0, 1.0 - (self.al / Ja) if Ja > 1e-16 else 0.0)
                    delta_alpha = C * dp * sign - gam * factor * alpha_old * dp
                    x = alpha_old + delta_alpha
                
                new_backstress[i] = x
            else:
                # Standard Chaboche for components 0-2 (analytical solution)
                numerator = alpha_old + C * dp * sign
                denominator = 1.0 + gam * dp
                new_backstress[i] = numerator / denominator
        
        return new_backstress

    def update_state_vars(self, state_vars, d_epsilon, sign):
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        dp = d_epsilon
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        return state_vars


class OhnoWangIIModelUniAxial(BaseMaterialModel):
    """
    Ohno-Wang II kinematic hardening model (1D version).
    """
    def __init__(self, isotropic_model, C, gamma, m):
        super().__init__(C=C, gamma=gamma, m=m)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.m = np.array(m)
        self.n_components = len(C)
        # Isotropic hardening model
        self.isotropic_model = isotropic_model

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Ohno-Wang II plastic modulus components.
        Returns total hardening modulus (h_kin + h_iso).
        """
        # 1D: n is just sign
        n = state_vars['sign']
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)

        # Kinematic hardening modulus
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_abs / denom) ** self.m[i] if denom != 0 else 0.0
            h_kin += self.C[i] - self.gamma[i] * pow_term * n * ai
        
        # Isotropic hardening modulus
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)

        return h_kin + h_iso

    def plastic_multiplier(self, f, E, backstress, state_vars):
        # Use the generic plastic modulus computation
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = E + h_total
        
        d_epsilon = f / denominator if denominator > 1e-12 else 0.0
        if d_epsilon < 0:
            d_epsilon = 0.0
            
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, sign, state_vars):
        new_backstress = backstress.copy()
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_abs / denom) ** self.m[i] if denom != 0 else 0.0
            delta_alpha = self.C[i] * d_epsilon * sign - self.gamma[i] * pow_term * d_epsilon * ai
            new_backstress[i] += delta_alpha
        return new_backstress

    def update_backstress_implicit(self, backstress, d_epsilon, sign):
        """
        Update backstress using implicit method (Backward Euler).
        Solves: alpha_new = alpha_old + C*dp*sign - gamma * (|alpha_new|/(C/gamma))^m * alpha_new * dp
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            alpha_old = backstress[i]
            C = self.C[i]
            gam = self.gamma[i]
            m = self.m[i]
            
            # RHS of equation: y = alpha_old + C * dp * sign
            # Equation to solve for x (alpha_new):
            # x * (1 + gam * dp * (|x| / (C/gam))^m) - RHS = 0
            
            RHS = alpha_old + C * dp * sign
            
            if gam == 0 or m == 0 or dp < 1e-16:
                # Linear case or no step
                new_backstress[i] = (RHS) / (1.0 + gam * dp) if dp > 0 else RHS
                continue

            threshold = C / gam
            RHS_mag = abs(RHS)
            RHS_sign = np.sign(RHS) if abs(RHS) > 0 else 1.0

            # Solve for magnitude y = |x|
            # y * (1 + gam * dp * (y / threshold)^m) - |RHS| = 0
            
            def residual_mag(y):
                 if y < 0: return -abs(RHS_mag) - y # penalize
                 term_pow = (y / threshold) ** m
                 return y * (1.0 + gam * dp * term_pow) - RHS_mag

            # Bracket: 
            # Lower = 0 (since y >= 0) -> Res = -|RHS| < 0
            # Upper = RHS_mag (since term_pow > 0, LHS > RHS_mag if y=RHS_mag) -> Res > 0?
            # Wait, if y=RHS_mag, term_pow > 0, so y*(1+...) > y = RHS_mag. 
            # So residual at RHS_mag is POSITIVE (or zero if term_pow=0).
            # So [0, RHS_mag] is a guaranteed bracket!
            
            # Optimization: If RHS is huge, upper bound is safe but we want accuracy.
            
            try:
                from scipy.optimize import brentq
                # Use brentq which is fast and robust
                y_sol = brentq(residual_mag, 0.0, RHS_mag, xtol=1e-12, maxiter=50)
                x = y_sol * RHS_sign
            except Exception:
                # Fallback to simple bisection if brentq fails (unlikely)
                low, high = 0.0, RHS_mag
                for _ in range(50):
                    mid = 0.5 * (low + high)
                    if residual_mag(mid) > 0:
                        high = mid
                    else:
                        low = mid
                x = mid * RHS_sign

            new_backstress[i] = x
            
        return new_backstress

    def update_state_vars(self, state_vars, d_epsilon, sign):
        """
        Update isotropic hardening state variables (R and p) using separate isotropic hardening model.
        """
        # Get current values from the solver state
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # d_epsilon is the increment of accumulated plastic strain (dp)
        dp = d_epsilon
        
        # Update isotropic hardening using separate model (explicit method)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        
        # Update state variables
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        
        return state_vars

class ChabocheModelMultiAxial(BaseMaterialModel):
    """
    Chaboche kinematic hardening model for multi-axial stress states.
    (Applying theoretically consistent formulation for 3D simulation)
    """
    def __init__(self, isotropic_model, C, gamma, E, nu=0.3):
        super().__init__(C=C, gamma=gamma)
        # For consistent 1D-3D behavior under uniaxial loading:
        # In 1D: dα = C*dp*sign - γ*α*dp
        # In 3D: dα = (2/3)*C*d_ε_p - γ*α*dp where d_ε_p = (3/2)*dp*n
        # Keep original C values - scaling was not effective
        self.C = np.array(C)  # Keep original C values
        self.gamma = np.array(gamma) # gamma parameter is consistent between 1D and 3D
        self.n_components = len(C)
        # Isotropic hardening model
        self.isotropic_model = isotropic_model
        # Store elastic properties
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Chaboche plastic modulus for multi-axial case.
        Returns the total hardening modulus considering tensor nature of multi-axial loading.
        """
        # Get the flow direction from state_vars if available
        flow_direction = state_vars.get('flow_direction', np.zeros((3, 3)))
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # Compute kinematic hardening modulus for multi-axial case
        h_kin = 0.0
        for i in range(self.n_components):
            # backstress should always have n_components entries (some may be zero tensors)
            alpha_i = backstress[i]
            
            # For multi-axial case, compute the directional hardening modulus
            # h_i = C_i - (2/3) * gamma_i * (n : alpha_i)
            # where n is the flow direction and alpha_i is the i-th backstress tensor
            
            # Compute the directional projection: n : alpha_i
            n_alpha_product = np.sum(flow_direction * alpha_i)
            h_i = self.C[i] - np.sqrt(3.0/2.0) * self.gamma[i] * n_alpha_product
            
            h_kin += h_i
        
        # Isotropic hardening modulus using separated model
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        
        return h_kin + h_iso

    def yield_function(self, trial_stress, total_backstress, sigma_y):
        eff_stress = trial_stress - total_backstress
        s = deviatoric(eff_stress)
        J2 = 0.5 * np.sum(s * s)
        f = np.sqrt(3 * J2) - sigma_y
        return f, s

    def plastic_multiplier(self, f, E, backstress, state_vars):
        """
        Compute the plastic multiplier for multi-axial case using explicit method.
        Enhanced with stability checks to prevent discontinuities during unloading.
        """
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        # For multi-axial case, use 3*G
        denominator = 3 * self.G + h_total
        
        # Enhanced stability check: ensure denominator is positive and reasonable
        if denominator <= 1e-12:
            # If plastic modulus is negative or too small, use a more conservative approach
            # This prevents numerical instabilities during unloading
            d_epsilon = 0.0
        else:
            d_epsilon = f / denominator
            
        # Additional check: ensure plastic multiplier is non-negative
        if d_epsilon < 0:
            d_epsilon = 0.0
            
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, flow_direction, state_vars):
        """
        Update backstress using explicit evolution equations for multi-axial case.
        """
        new_backstress = [np.zeros((3, 3)) for _ in backstress]
        d_epsilon_p = np.sqrt(3. / 2.) * d_epsilon * flow_direction
        
        for i in range(self.n_components):
            # Explicit update: dα_i = (2/3)*C_i*d_ε_p - γ_i*α_i*dp
            delta_alpha = (2.0 / 3.0) * self.C[i] * d_epsilon_p - self.gamma[i] * backstress[i] * d_epsilon
            new_backstress[i] = backstress[i] + delta_alpha
        
        return new_backstress

    def update_backstress_implicit(self, backstress_tensors_old, dp, flow_direction):
        new_backstress = [np.zeros((3, 3)) for _ in backstress_tensors_old]
        d_epsilon_p = np.sqrt(3. / 2.) * dp * flow_direction
        for i in range(self.n_components):
            # Standard 3D Chaboche implicit update rule
            numerator = backstress_tensors_old[i] + (2. / 3.) * self.C[i] * d_epsilon_p
            # For 3D, gamma term should be: γ*||α||*dp, but for implicit form it's γ*dp
            denominator = 1.0 + self.gamma[i] * dp
            new_backstress[i] = numerator / denominator
        return new_backstress

    def update_state_vars(self, state_vars, dp, sign=None):
        """
        Update isotropic hardening state variables (R and p) using separate isotropic hardening model.
        """
        # Get current values from the solver state
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # Update isotropic hardening using separate model (explicit method)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        
        # Update state variables
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        
        return state_vars

    def update_state_vars_implicit(self, R_old, p_old, dp):
        """
        Update isotropic hardening state variables (R and p) using separate isotropic hardening model (implicit method).
        Uses legacy behavior (no saturation bounds) to match original multi-axial implementation.
        """
        # Use the legacy implicit update method to match original behavior
        R_new, p_new = self.isotropic_model.update_implicit(R_old, p_old, dp)
        return R_new, p_new 

class OhnoWangIIModelMultiAxial(BaseMaterialModel):
    """
    Ohno-Wang II kinematic hardening model for multi-axial stress states (fully implicit version).
    Implements the correct tensorial evolution law and plastic modulus as per the user's formula.
    """
    def __init__(self, isotropic_model, C, gamma, m, E, nu=0.3):
        super().__init__(C=C, gamma=gamma, m=m)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.m = np.array(m)
        self.n_components = len(C)
        self.isotropic_model = isotropic_model
        # Store elastic properties
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Ohno-Wang II plastic modulus for multi-axial case.
        h = sum_i [ C_i - sqrt(3/2) * gamma_i * (|a_i|/(C_i/gamma_i))^m * (n : a_i) ] + h_iso
        """
        flow_direction = state_vars.get('flow_direction', np.zeros((3, 3)))
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            # a_bar = sqrt(3/2 * a:a) = sqrt(3/2) * ||a|| (Mises-equivalent backstress)
            ai_bar = np.sqrt(3.0 / 2.0) * ai_norm if ai_norm > 1e-16 else 0.0
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_bar / denom) ** self.m[i] if denom != 0 else 0.0
            n_dot_ai = np.sum(flow_direction * ai)
            h_i = self.C[i] - np.sqrt(3.0/2.0) * self.gamma[i] * pow_term * n_dot_ai
            h_kin += h_i
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        return h_kin + h_iso

    def yield_function(self, trial_stress, total_backstress, sigma_y):
        eff_stress = trial_stress - total_backstress
        s = deviatoric(eff_stress)
        J2 = 0.5 * np.sum(s * s)
        f = np.sqrt(3 * J2) - sigma_y
        return f, s

    def plastic_multiplier(self, f, E, backstress, state_vars):
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        # For multi-axial case, use 3*G
        denominator = 3 * self.G + h_total
        if denominator <= 1e-12:
            d_epsilon = 0.0
        else:
            d_epsilon = f / denominator
        if d_epsilon < 0:
            d_epsilon = 0.0
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, flow_direction, state_vars):
        """
        Update backstress using Ohno-Wang II evolution law (explicit, multi-axial).
        dα_i = (2/3)*C_i*dε_p - γ_i*(|a_i|/(C_i/γ_i))^m * dp * a_i
        """
        new_backstress = [np.zeros((3, 3)) for _ in backstress]
        d_epsilon_p = np.sqrt(3. / 2.) * d_epsilon * flow_direction
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            ai_bar = np.sqrt(3.0 / 2.0) * ai_norm if ai_norm > 1e-16 else 0.0
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_bar / denom) ** self.m[i] if denom != 0 else 0.0
            delta_alpha = (2.0 / 3.0) * self.C[i] * d_epsilon_p - self.gamma[i] * pow_term * d_epsilon * ai
            new_backstress[i] = ai + delta_alpha
        return new_backstress

    def update_backstress_implicit(self, backstress_tensors_old, dp, flow_direction):
        """
        Implicit update for Ohno-Wang II backstress (multi-axial, fully implicit step).
        Backward Euler: a_i_new = (a_i_old + (2/3)*C_i*dε_p) / (1 + γ_i*pow_term*dp)
        """
        new_backstress = [np.zeros((3, 3)) for _ in backstress_tensors_old]
        d_epsilon_p = np.sqrt(3. / 2.) * dp * flow_direction
        for i in range(self.n_components):
            ai_old = backstress_tensors_old[i]
            ai_norm = np.linalg.norm(ai_old)
            ai_bar = np.sqrt(3.0 / 2.0) * ai_norm if ai_norm > 1e-16 else 0.0
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_bar / denom) ** self.m[i] if denom != 0 else 0.0
            denominator_implicit = 1.0 + self.gamma[i] * pow_term * dp
            numerator_implicit = ai_old + (2.0 / 3.0) * self.C[i] * d_epsilon_p
            new_backstress[i] = numerator_implicit / denominator_implicit
        return new_backstress

    def update_state_vars(self, state_vars, dp, sign=None):
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        return state_vars

    def update_state_vars_implicit(self, R_old, p_old, dp):
        R_new, p_new = self.isotropic_model.update_implicit(R_old, p_old, dp)
        return R_new, p_new 

class AbdelKarimOhnoModelMultiAxial(BaseMaterialModel):
    """
    Abdel-Karim & Ohno kinematic hardening model for multi-axial stress states (explicit version).
    Implements the provided tensorial evolution law and plastic modulus.
    """
    def __init__(self, isotropic_model, C, gamma, mu, E, nu=0.3):
        super().__init__(C=C, gamma=gamma, mu=mu)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.mu = np.array(mu)
        self.n_components = len(C)
        self.isotropic_model = isotropic_model
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Abdel-Karim & Ohno plastic modulus for multi-axial case.
        Only the first bracket is Macaulay (positive part), the second is a normal scalar product.
        """
        flow_direction = state_vars.get('flow_direction', np.zeros((3, 3)))
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            
            # Always include the hardening term (2/3)*C_i
            h_i = self.C[i]
            
            # Include recovery term only when backstress exists
            if ai_norm > 1e-14:
                n_dot_ai = np.sum(flow_direction * ai)
                ai_bar = np.sqrt(3.0/2.0) * ai_norm
                ri = self.C[i]/self.gamma[i]
                gi = ai_bar**2 - ri**2
                H_gi = 1.0 if gi > 0 else 0.0
                n_dot_ai_unit = n_dot_ai / ai_bar
                macaulay = max(0.0, n_dot_ai_unit)
                # Recovery term
                h_i -= 3/2* self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * ai_bar * macaulay * n_dot_ai_unit
            
            h_kin += h_i
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        return h_kin + h_iso

    def yield_function(self, trial_stress, total_backstress, sigma_y):
        eff_stress = trial_stress - total_backstress
        s = deviatoric(eff_stress)
        J2 = 0.5 * np.sum(s * s)
        f = np.sqrt(3 * J2) - sigma_y
        return f, s

    def plastic_multiplier(self, f, E, backstress, state_vars):
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = 3 * self.G + h_total
        if denominator <= 1e-12:
            d_epsilon = 0.0
        else:
            d_epsilon = f / denominator
        if d_epsilon < 0:
            d_epsilon = 0.0
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, flow_direction, state_vars):
        """
        Update backstress using explicit Abdel-Karim & Ohno evolution law (multi-axial).
        """
        new_backstress = [np.zeros((3, 3)) for _ in backstress]
        d_epsilon_p = np.sqrt(3. / 2.) * d_epsilon * flow_direction
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            
            # Always apply the first term (hardening)
            delta_alpha = (2.0 / 3.0) * self.C[i] * d_epsilon_p
            
            # Apply the second term (recovery) only if backstress is not zero
            if ai_norm > 1e-14:
                n_dot_ai = np.sum(flow_direction * ai)
                ai_bar = np.sqrt(3.0/2.0) * ai_norm
                ri = self.C[i]/self.gamma[i]
                gi = ai_bar**2 - ri**2
                H_gi = 1.0 if gi > 0 else 0.0
                n_dot_ai_unit = n_dot_ai / ai_bar
                bracket = max(0.0, n_dot_ai_unit)
                # Recovery term
                delta_alpha -= self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * bracket * d_epsilon * ai
            
            new_backstress[i] = ai + delta_alpha
        return new_backstress

    def update_state_vars(self, state_vars, dp, sign=None):
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        return state_vars

    def update_state_vars_implicit(self, R_old, p_old, dp):
        R_new, p_new = self.isotropic_model.update_implicit(R_old, p_old, dp)
        return R_new, p_new 

class AbdelKarimOhnoModelUniAxial(BaseMaterialModel):
    """
    Abdel-Karim & Ohno kinematic hardening model (1D uniaxial version).
    Derived from multi-axial version by simplifying for uniaxial case.
    
    Key simplifications for uniaxial:
    - Flow direction n becomes sign (±1)
    - Backstress α_i is scalar (not tensor)
    - ᾱ_i = |α_i| (simplified from √(3/2) * ||α_i||)
    - n:α_i = sign * α_i (scalar product)
    """
    def __init__(self, isotropic_model, C, gamma, mu):
        super().__init__(C=C, gamma=gamma, mu=mu)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.mu = np.array(mu)
        self.n_components = len(C)
        # Isotropic hardening model
        self.isotropic_model = isotropic_model

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Abdel-Karim & Ohno plastic modulus for uniaxial case.
        Returns total hardening modulus (h_kin + h_iso).
        """
        # 1D: n is just sign
        sign = state_vars['sign']
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            
            # Always include the hardening term C_i
            h_i = self.C[i]
            
            # Include recovery term only when backstress exists
            if ai_abs > 1e-14:
                # AKO recovery term with Macaulay bracket - strictly following multi-axial theory
                # Multi-axial: h_i = C_i - (3/2) * γ_i * (μ_i + H_gi * (1 - μ_i)) * ᾱ_i * <n:α_i/ᾱ_i>_+ * (n:α_i/ᾱ_i)
                # Uniaxial derivation:
                # - ᾱ_i = √(3/2) * |α_i| (consistent with multi-axial)
                ai_bar = np.sqrt(3.0/2.0) * ai_abs
                # - r_i = C_i / γ_i
                ri = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1e10
                # - g_i = ᾱ_i² - r_i²
                gi = ai_bar**2 - ri**2
                # - H_gi = 1 if g_i > 0 else 0
                H_gi = 1.0 if gi > 0 else 0.0
                # - n:α_i = sign * α_i (scalar product in uniaxial)
                n_dot_ai = sign * ai
                # - n:α_i/ᾱ_i = (sign * α_i) / (√(3/2) * |α_i|) = sign * sign(α_i) / √(3/2)
                ai_sign = np.sign(ai) if ai_abs > 0 else 1.0
                n_dot_ai_unit = n_dot_ai / ai_bar  # = sign * sign(α_i) / √(3/2)
                # - Macaulay bracket: <n:α_i/ᾱ_i>_+ = max(0, n:α_i/ᾱ_i)
                macaulay = max(0.0, n_dot_ai_unit)
                # Recovery term (following multi-axial formula exactly)
                recovery_coeff = (3.0/2.0) * self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * ai_bar * macaulay * n_dot_ai_unit
                h_i -= recovery_coeff
            
            h_kin += h_i
        
        # Isotropic hardening modulus
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        
        return h_kin + h_iso

    def plastic_multiplier(self, f, E, backstress, state_vars):
        """Compute the plastic multiplier for uniaxial case."""
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = E + h_total
        
        d_epsilon = f / denominator if denominator > 1e-12 else 0.0
        if d_epsilon < 0:
            d_epsilon = 0.0
            
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, sign, state_vars):
        """
        Update backstress using explicit Abdel-Karim & Ohno evolution law (uniaxial).
        Explicit method: uses current (old) backstress values.
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            
            # Always apply the first term (hardening)
            delta_alpha = self.C[i] * dp * sign
            
            # Apply the second term (recovery) only if backstress is not zero
            if ai_abs > 1e-14:
                # AKO recovery term with Macaulay bracket - strictly following multi-axial theory
                # Multi-axial: dα_i = (2/3) * C_i * dε_p - γ_i * (μ_i + H_gi * (1 - μ_i)) * <n:α_i/ᾱ_i>_+ * dp * α_i
                # Uniaxial derivation (same as in compute_plastic_modulus):
                ai_bar = np.sqrt(3.0/2.0) * ai_abs
                ri = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1e10
                gi = ai_bar**2 - ri**2
                H_gi = 1.0 if gi > 0 else 0.0
                n_dot_ai = sign * ai
                ai_sign = np.sign(ai) if ai_abs > 0 else 1.0
                n_dot_ai_unit = n_dot_ai / ai_bar  # = sign * sign(α_i) / √(3/2)
                bracket = max(0.0, n_dot_ai_unit)  # Macaulay bracket
                # Recovery term (following multi-axial formula exactly)
                recovery_term = self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * bracket * dp * ai
                delta_alpha -= recovery_term
            
            new_backstress[i] += delta_alpha
            
        return new_backstress

    def update_backstress_implicit(self, backstress, d_epsilon, sign):
        """
        Update backstress using implicit method (Backward Euler) for uniaxial case.
        
        Implicit equation to solve for α_new:
        α_new = α_old + C * dp * sign - γ * (μ + H_gi * (1-μ)) * <n:α_new/|α_new|>_+ * dp * α_new
        
        This is non-linear and requires numerical solution.
        We solve for the magnitude |α_new| first, then recover the sign.
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            alpha_old = backstress[i]
            C = self.C[i]
            gam = self.gamma[i]
            mu = self.mu[i]
            
            # RHS of equation: RHS = alpha_old + C * dp * sign
            RHS = alpha_old + C * dp * sign
            RHS_mag = abs(RHS)
            RHS_sign = np.sign(RHS) if abs(RHS) > 1e-16 else (1.0 if sign > 0 else -1.0)
            
            # If no recovery (gamma = 0) or no step, solution is trivial
            if gam == 0 or dp < 1e-16:
                new_backstress[i] = RHS
                continue
            
            # AKO implicit update - strictly following multi-axial theory
            # Implicit equation: α_new = α_old + C * dp * sign - γ * (μ + H_gi * (1-μ)) * <n:α_new/ᾱ_new>_+ * dp * α_new
            # This is non-linear because H_gi depends on |α_new| and Macaulay bracket depends on direction
            # Define parameters
            ri = C / gam if gam > 0 else 1e10
            
            # Solve for magnitude y = |α_new| using the same formula as multi-axial
            # In uniaxial: n:α_new = sign * α_new, ᾱ_new = √(3/2) * |α_new|
            # The implicit equation in terms of magnitude:
            # y * [1 + γ * dp * (μ + H_gi * (1-μ)) * <sign * sign(α_new)>_+ * (sign * sign(α_new)) / √(3/2)] = |RHS|
            # where:
            #   - sign(α_new) = sign(RHS) (we assume α_new has same sign as RHS)
            #   - <sign * sign(RHS)>_+ = max(0, sign * sign(RHS))
            #   - g_i = (√(3/2) * y)² - r_i² = (3/2) * y² - r_i²
            #   - H_gi = 1 if g_i > 0 else 0
            
            # Define residual function for magnitude following multi-axial formula exactly
            # Implicit equation: α_new = α_old + C * dp * sign - γ * (μ + H_gi*(1-μ)) * <n:α_new/ᾱ_new>_+ * dp * α_new
            # Rearranging: α_new + γ * (μ + H_gi*(1-μ)) * <n:α_new/ᾱ_new>_+ * dp * α_new = α_old + C * dp * sign
            # In terms of magnitude y = |α_new|: y * (1 + recovery_coeff * dp) = |RHS|
            # where recovery_coeff = γ * (μ + H_gi*(1-μ)) * <n:α_new/ᾱ_new>_+
            def residual_mag(y):
                if y < 0:
                    return -abs(RHS_mag) - y  # Penalize negative
                
                # Compute ᾱ_new = √(3/2) * y (consistent with multi-axial)
                ai_bar_new = np.sqrt(3.0/2.0) * y
                
                # Compute g_i = ᾱ_new² - r_i²
                gi = ai_bar_new**2 - ri**2
                H_gi = 1.0 if gi > 0 else 0.0
                
                # In uniaxial: α_new = sign(α_new) * y = RHS_sign * y
                # n:α_new = sign * α_new = sign * RHS_sign * y
                # n:α_new/ᾱ_new = (sign * RHS_sign * y) / (√(3/2) * y) = sign * RHS_sign / √(3/2)
                n_dot_ai_unit_new = sign * RHS_sign / np.sqrt(3.0/2.0)
                
                # Macaulay bracket: <n:α_new/ᾱ_new>_+
                macaulay = max(0.0, n_dot_ai_unit_new)
                
                # Recovery coefficient following multi-axial formula exactly:
                # Same as explicit update: γ * (μ + H_gi*(1-μ)) * bracket
                recovery_coeff = gam * (mu + H_gi * (1.0 - mu)) * macaulay
                
                # Implicit equation: y * (1 + recovery_coeff * dp) = |RHS|
                return y * (1.0 + recovery_coeff * dp) - RHS_mag
            
            # Bracket search: [0, |RHS|] should bracket the root
            # At y=0: residual = -|RHS| < 0
            # At y=|RHS|: if recovery_coeff > 0, residual > 0 (or zero)
            # We need to find upper bound where residual > 0
            lower = 0.0
            upper = RHS_mag
            
            # Check if upper bound gives positive residual
            if residual_mag(upper) <= 0:
                # Need to expand upper bound
                upper = RHS_mag * 2.0
                max_iter = 10
                for _ in range(max_iter):
                    if residual_mag(upper) > 0:
                        break
                    upper *= 2.0
            
            # Solve using brentq
            try:
                from scipy.optimize import brentq
                y_sol = brentq(residual_mag, lower, upper, xtol=1e-12, maxiter=50)
                x = y_sol * RHS_sign
            except Exception as e:
                # Fallback: use simple iteration or explicit update
                # For robustness, fall back to explicit update
                # Alternatively, try bisection
                try:
                    low, high = lower, upper
                    for _ in range(50):
                        mid = 0.5 * (low + high)
                        res_mid = residual_mag(mid)
                        if abs(res_mid) < 1e-12:
                            x = mid * RHS_sign
                            break
                        elif res_mid > 0:
                            high = mid
                        else:
                            low = mid
                    else:
                        # If bisection didn't converge, use explicit update as fallback
                        # This should not happen in normal cases, but provides robustness
                        delta_alpha = C * dp * sign
                        if abs(alpha_old) > 1e-14:
                            # Use explicit AKO update formula (strictly following theory)
                            ai_bar = np.sqrt(3.0/2.0) * abs(alpha_old)
                            ri_val = C / gam if gam > 0 else 1e10
                            gi = ai_bar**2 - ri_val**2
                            H_gi = 1.0 if gi > 0 else 0.0
                            n_dot_ai = sign * alpha_old
                            ai_sign = np.sign(alpha_old)
                            n_dot_ai_unit = n_dot_ai / ai_bar
                            macaulay = max(0.0, n_dot_ai_unit)
                            recovery_term = gam * (mu + H_gi * (1.0 - mu)) * macaulay * dp * alpha_old
                            delta_alpha -= recovery_term
                        x = alpha_old + delta_alpha
                except Exception:
                    # Final fallback: explicit update (strictly following AKO theory, no special cases)
                    delta_alpha = C * dp * sign
                    if abs(alpha_old) > 1e-14:
                        # Use explicit AKO update formula (strictly following theory)
                        ai_bar = np.sqrt(3.0/2.0) * abs(alpha_old)
                        ri_val = C / gam if gam > 0 else 1e10
                        gi = ai_bar**2 - ri_val**2
                        H_gi = 1.0 if gi > 0 else 0.0
                        n_dot_ai = sign * alpha_old
                        ai_sign = np.sign(alpha_old)
                        n_dot_ai_unit = n_dot_ai / ai_bar
                        macaulay = max(0.0, n_dot_ai_unit)
                        recovery_term = gam * (mu + H_gi * (1.0 - mu)) * macaulay * dp * alpha_old
                        delta_alpha -= recovery_term
                    x = alpha_old + delta_alpha
            
            new_backstress[i] = x
            
        return new_backstress

    def update_state_vars(self, state_vars, d_epsilon, sign):
        """
        Update isotropic hardening state variables (R and p) using separate isotropic hardening model.
        """
        # Get current values from the solver state
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # d_epsilon is the increment of accumulated plastic strain (dp)
        dp = d_epsilon
        
        # Update isotropic hardening using separate model (explicit method)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        
        # Update state variables
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        
        return state_vars

class KangModelUniAxial(BaseMaterialModel):
    """
    Kang kinematic hardening model (1D uniaxial version).
    Derived from multi-axial version by simplifying for uniaxial case.
    
    Key differences from AKO:
    - No Macaulay bracket: all plastic directions contribute to recovery
    - Structure identical to AKO, only recovery term differs
    
    Key simplifications for uniaxial:
    - Flow direction n becomes sign (±1)
    - Backstress α_i is scalar (not tensor)
    - ᾱ_i = √(3/2) * |α_i| (consistent with multi-axial)
    - n:α_i = sign * α_i (scalar product)
    """
    def __init__(self, isotropic_model, C, gamma, mu):
        super().__init__(C=C, gamma=gamma, mu=mu)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.mu = np.array(mu)
        self.n_components = len(C)
        # Isotropic hardening model
        self.isotropic_model = isotropic_model

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Kang plastic modulus for uniaxial case.
        No Macaulay bracket: all directions contribute to recovery.
        Returns total hardening modulus (h_kin + h_iso).
        """
        # 1D: n is just sign
        sign = state_vars['sign']
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            
            # Always include the hardening term C_i
            h_i = self.C[i]
            
            # Include recovery term only when backstress exists
            if ai_abs > 1e-14:
                # Kang recovery term - strictly following multi-axial theory
                # Multi-axial: h_i = C_i - √(3/2) * γ_i * (μ_i + H_gi * (1 - μ_i)) * ᾱ_i * (n:α_i/ᾱ_i)
                # No Macaulay bracket: all directions contribute
                # Uniaxial derivation:
                # - ᾱ_i = √(3/2) * |α_i| (consistent with multi-axial)
                ai_bar = ai_abs  # Corrected: removed √(3/2) factor
                # - r_i = C_i / γ_i
                ri = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1e10
                # - g_i = ᾱ_i² - r_i²
                gi = ai_bar**2 - ri**2
                # - H_gi = 1 if g_i > 0 else 0
                H_gi = 1.0 if gi > 0 else 0.0
                # - n:α_i = sign * α_i (scalar product in uniaxial)
                n_dot_ai = sign * ai
                # - n:α_i/ᾱ_i = (sign * α_i) / (√(3/2) * |α_i|) = sign * sign(α_i) / √(3/2)
                ai_sign = np.sign(ai) if ai_abs > 0 else 1.0
                n_dot_ai_unit = n_dot_ai / ai_bar  # = sign * sign(α_i) / √(3/2)
                # Recovery term (no Macaulay bracket - all directions contribute)
                recovery_coeff = self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * sign * ai
                h_i -= recovery_coeff
            
            h_kin += h_i
        
        # Isotropic hardening modulus
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        
        return h_kin + h_iso

    def plastic_multiplier(self, f, E, backstress, state_vars):
        """Compute the plastic multiplier for uniaxial case."""
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = E + h_total
        
        d_epsilon = f / denominator if denominator > 1e-12 else 0.0
        if d_epsilon < 0:
            d_epsilon = 0.0
            
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, sign, state_vars):
        """
        Update backstress using explicit Kang evolution law (uniaxial).
        Explicit method: uses current (old) backstress values.
        No Macaulay bracket: all directions contribute.
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            
            # Always apply the first term (hardening)
            delta_alpha = self.C[i] * dp * sign
            
            # Apply the second term (recovery) only if backstress is not zero
            if ai_abs > 1e-14:
                # Kang recovery term - strictly following multi-axial theory
                # Multi-axial: dα_i = (2/3) * C_i * dε_p - γ_i * (μ_i + H_gi * (1 - μ_i)) * dp * α_i
                # No Macaulay bracket: all directions contribute
                # Uniaxial derivation (same as in compute_plastic_modulus):
                ai_bar = ai_abs  # Corrected: removed √(3/2) factor
                ri = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1e10
                gi = ai_bar**2 - ri**2
                H_gi = 1.0 if gi > 0 else 0.0
                # Recovery term (no Macaulay bracket - all directions contribute)
                recovery_term = self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * dp * ai
                delta_alpha -= recovery_term
            
            new_backstress[i] += delta_alpha
            
        return new_backstress

    def update_backstress_implicit(self, backstress, d_epsilon, sign):
        """
        Update backstress using implicit method (Backward Euler) for uniaxial case.
        
        Implicit equation to solve for α_new:
        α_new = α_old + C * dp * sign - γ * (μ + H_gi * (1-μ)) * dp * α_new
        
        This is non-linear because H_gi depends on |α_new|.
        We solve for the magnitude |α_new| first, then recover the sign.
        No Macaulay bracket: all directions contribute.
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            alpha_old = backstress[i]
            C = self.C[i]
            gam = self.gamma[i]
            mu = self.mu[i]
            
            # RHS of equation: RHS = alpha_old + C * dp * sign
            RHS = alpha_old + C * dp * sign
            RHS_mag = abs(RHS)
            RHS_sign = np.sign(RHS) if abs(RHS) > 1e-16 else (1.0 if sign > 0 else -1.0)
            
            # If no recovery (gamma = 0) or no step, solution is trivial
            if gam == 0 or dp < 1e-16:
                new_backstress[i] = RHS
                continue
            
            # Kang implicit update - strictly following multi-axial theory
            # No Macaulay bracket: all directions contribute
            # Define parameters
            ri = C / gam if gam > 0 else 1e10
            
            # CORRECTED: Solve for α_new (with sign), not just magnitude
            # Implicit equation: α_new * (1 + γ*(μ + H_gi*(1-μ))*dp) = RHS
            # where H_gi depends on |α_new|
            
            def residual_signed(x):
                """Residual: x * (1 + recovery_coeff * dp) - RHS"""
                abs_x = abs(x)
                gi = abs_x**2 - ri**2
                H_gi = 1.0 if gi > 0 else 0.0
                recovery_coeff = gam * (mu + H_gi * (1.0 - mu))
                return x * (1.0 + recovery_coeff * dp) - RHS
            
            # Initial guess: explicit update
            x_explicit = alpha_old + C * dp * sign
            if abs(alpha_old) > 1e-14:
                ai_bar = abs(alpha_old)
                gi = ai_bar**2 - ri**2
                H_gi = 1.0 if gi > 0 else 0.0
                recovery_term = gam * (mu + H_gi * (1.0 - mu)) * dp * alpha_old
                x_explicit -= recovery_term
            
            # Determine bracket
            if RHS > 0:
                lower = 0.0
                upper = max(RHS * 1.5, x_explicit * 1.5) if x_explicit > 0 else RHS * 1.5
            else:
                lower = min(RHS * 1.5, x_explicit * 1.5) if x_explicit < 0 else RHS * 1.5
                upper = 0.0
            
            # Solve with brentq
            try:
                from scipy.optimize import brentq, newton
                res_lower = residual_signed(lower)
                res_upper = residual_signed(upper)
                
                if res_lower * res_upper > 0:
                    try:
                        x = newton(residual_signed, x_explicit, tol=1e-12, maxiter=50)
                    except:
                        x = x_explicit
                else:
                    x = brentq(residual_signed, lower, upper, xtol=1e-12, maxiter=100)
            except Exception as e:
                x = x_explicit
            
            new_backstress[i] = x
            
        return new_backstress

    def update_state_vars(self, state_vars, d_epsilon, sign):
        """
        Update isotropic hardening state variables (R and p) using separate isotropic hardening model.
        """
        # Get current values from the solver state
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # d_epsilon is the increment of accumulated plastic strain (dp)
        dp = d_epsilon
        
        # Update isotropic hardening using separate model (explicit method)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        
        # Update state variables
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        
        return state_vars

class KangModelMultiAxial(BaseMaterialModel):
    """
    Kang kinematic hardening model for multi-axial stress states (explicit version).
    Structure identical to AKO, only backstress and plastic modulus formulas differ (no Macaulay bracket).
    """
    def __init__(self, isotropic_model, C, gamma, mu, E, nu=0.3):
        super().__init__(C=C, gamma=gamma, mu=mu)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.mu = np.array(mu)
        self.n_components = len(C)
        self.isotropic_model = isotropic_model
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))

    def update_backstress(self, backstress, d_epsilon, flow_direction, state_vars):
        """
        Update backstress using explicit Kang evolution law (multi-axial).
        No Macaulay bracket: all plastic directions contribute to recovery.
        """
        new_backstress = [np.zeros((3, 3)) for _ in backstress]
        d_epsilon_p = np.sqrt(3. / 2.) * d_epsilon * flow_direction
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            n_dot_ai = np.sum(flow_direction * ai)
            ai_bar = np.sqrt(3.0/2.0) * ai_norm
            ri = self.C[i]/self.gamma[i]
            gi = ai_bar**2 - ri**2
            H_gi = 1.0 if gi > 0 else 0.0
            n_dot_ai_unit = n_dot_ai / ai_bar if ai_norm > 1e-14 else 0.0
            # Kang model: no Macaulay, all directions contribute
            delta_alpha = (2.0 / 3.0) * self.C[i] * d_epsilon_p \
                - self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * d_epsilon * ai
            new_backstress[i] = ai + delta_alpha
        return new_backstress

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Kang plastic modulus for multi-axial case.
        No Macaulay bracket: all plastic directions contribute.
        """
        flow_direction = state_vars.get('flow_direction', np.zeros((3, 3)))
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            n_dot_ai = np.sum(flow_direction * ai)
            ai_bar = np.sqrt(3.0/2.0) * ai_norm
            ri = self.C[i]/self.gamma[i]
            gi = ai_bar**2 - ri**2
            H_gi = 1.0 if gi > 0 else 0.0
            n_dot_ai_unit = n_dot_ai / ai_bar if ai_norm > 1e-14 else 0.0
            # Kang model: no Macaulay, all directions contribute
            h_kin +=  self.C[i] - np.sqrt(3/2)*self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * ai_bar * n_dot_ai_unit
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        return h_kin + h_iso

    # All other methods (yield_function, plastic_multiplier, etc.) are identical to AKO
    def yield_function(self, trial_stress, total_backstress, sigma_y):
        eff_stress = trial_stress - total_backstress
        s = deviatoric(eff_stress)
        J2 = 0.5 * np.sum(s * s)
        f = np.sqrt(3 * J2) - sigma_y
        return f, s

    def plastic_multiplier(self, f, E, backstress, state_vars):
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = 3 * self.G + h_total
        if denominator <= 1e-12:
            d_epsilon = 0.0
        else:
            d_epsilon = f / denominator
        if d_epsilon < 0:
            d_epsilon = 0.0
        return d_epsilon

    def update_state_vars(self, state_vars, dp, sign=None):
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        return state_vars

    def update_state_vars_implicit(self, R_old, p_old, dp):
        R_new, p_new = self.isotropic_model.update_implicit(R_old, p_old, dp)
        return R_new, p_new


class RenModelUniAxial(BaseMaterialModel):
    """
    Ren kinematic hardening model (1D uniaxial version).
    Derived from multi-axial version by simplifying for uniaxial case.
    
    Key differences from AKO:
    - No Macaulay bracket: all plastic directions contribute to recovery
    - Structure identical to AKO, only recovery term differs
    - Uses sigmoidal approximation for Heaviside function: H(gi) = 1 / (1 + exp(-k * gi))
    
    Key simplifications for uniaxial:
    - Flow direction n becomes sign (±1)
    - Backstress α_i is scalar (not tensor)
    - ᾱ_i = √(3/2) * |α_i| (consistent with multi-axial)
    - n:α_i = sign * α_i (scalar product)
    
    Parameters:
    - k: Steepness parameter for sigmoidal Heaviside function (larger k = sharper transition)
    """
    def __init__(self, isotropic_model, C, gamma, mu, k=1000.0):
        super().__init__(C=C, gamma=gamma, mu=mu)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.mu = np.array(mu)
        self.k = float(k)  # Steepness parameter for sigmoidal Heaviside function
        self.n_components = len(C)
        # Isotropic hardening model
        self.isotropic_model = isotropic_model
    
    def heaviside_sigmoidal(self, gi):
        """
        Compute sigmoidal approximation of Heaviside function.
        
        Uses logistic function: H(gi) = 1 / (1 + exp(-k * gi))
        
        Args:
            gi: Input value (typically gi = ᾱ_i² - r_i²)
        
        Returns:
            Smooth approximation of Heaviside step function
            - When gi >> 0: H ≈ 1
            - When gi << 0: H ≈ 0
            - When gi = 0: H = 0.5
            - As k → ∞: approaches true Heaviside step function
        """
        # Logistic function: 1 / (1 + exp(-k * gi))
        # For numerical stability, use: 1 / (1 + exp(-k * gi))
        return 1.0 / (1.0 + np.exp(-self.k * gi))

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Ren plastic modulus for uniaxial case.
        No Macaulay bracket: all directions contribute to recovery.
        Returns total hardening modulus (h_kin + h_iso).
        """
        # 1D: n is just sign
        sign = state_vars['sign']
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            
            # Always include the hardening term C_i
            h_i = self.C[i]
            
            # Include recovery term only when backstress exists
            if ai_abs > 1e-14:
                # Ren recovery term - uniaxial derivation from multi-axial theory
                # Multi-axial: h_i = C_i - √(3/2) * γ_i * (μ_i + H_gi * (1 - μ_i)) * ᾱ_i * (n:α_i/ᾱ_i)
                # No Macaulay bracket: all directions contribute
                # Uniaxial derivation:
                # - ᾱ_i = |α_i| (simplified from √(3/2) * ||α_i|| in multi-axial)
                ai_bar = ai_abs
                # - r_i = C_i / γ_i (same as multi-axial)
                ri = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1e10
                # - g_i = ᾱ_i² - r_i²
                gi = ai_bar**2 - ri**2
                # - H_gi: Sigmoidal approximation of Heaviside function
                H_gi = self.heaviside_sigmoidal(gi)
                # - n:α_i = sign * α_i (scalar product in uniaxial)
                n_dot_ai = sign * ai
                # - n:α_i/ᾱ_i = (sign * α_i) / |α_i| = sign * sign(α_i)
                ai_sign = np.sign(ai) if ai_abs > 0 else 1.0
                n_dot_ai_unit = n_dot_ai / ai_bar  # = sign * sign(α_i)
                # Recovery term (no Macaulay bracket - all directions contribute)
                # In uniaxial: simplified from multi-axial
                # Multi-axial: h_i = C_i - √(3/2) * γ_i * (μ_i + H_gi * (1 - μ_i)) * ᾱ_i * (n:α_i/ᾱ_i)
                # Uniaxial: h_i = C_i - γ_i * (μ_i + H_gi * (1 - μ_i)) * sign * α_i
                recovery_coeff = self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * sign * ai
                h_i -= recovery_coeff
            
            h_kin += h_i
        
        # Isotropic hardening modulus
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        
        return h_kin + h_iso

    def plastic_multiplier(self, f, E, backstress, state_vars):
        """Compute the plastic multiplier for uniaxial case."""
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = E + h_total
        
        d_epsilon = f / denominator if denominator > 1e-12 else 0.0
        if d_epsilon < 0:
            d_epsilon = 0.0
            
        return d_epsilon

    def update_backstress(self, backstress, d_epsilon, sign, state_vars):
        """
        Update backstress using explicit Ren evolution law (uniaxial).
        Explicit method: uses current (old) backstress values.
        No Macaulay bracket: all directions contribute.
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            ai = backstress[i]
            ai_abs = np.abs(ai)
            
            # Always apply the first term (hardening)
            delta_alpha = self.C[i] * dp * sign
            
            # Apply the second term (recovery) only if backstress is not zero
            if ai_abs > 1e-14:
                # Ren recovery term - uniaxial derivation from multi-axial theory
                # Multi-axial: dα_i = (2/3) * C_i * dε_p - γ_i * (μ_i + H_gi * (1 - μ_i)) * dp * α_i
                # No Macaulay bracket: all directions contribute
                # Uniaxial derivation:
                # - ᾱ_i = |α_i| (simplified from √(3/2) * ||α_i|| in multi-axial)
                ai_bar = ai_abs
                # - r_i = C_i / γ_i (same as multi-axial)
                ri = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1e10
                # - g_i = ᾱ_i² - r_i²
                gi = ai_bar**2 - ri**2
                H_gi = self.heaviside_sigmoidal(gi)
                # Recovery term (no Macaulay bracket - all directions contribute)
                # In uniaxial: same as multi-axial, recovery_term = γ_i * (μ_i + H_gi * (1 - μ_i)) * dp * α_i
                recovery_term = self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * dp * ai
                delta_alpha -= recovery_term
            
            new_backstress[i] += delta_alpha
            
        return new_backstress

    def update_backstress_implicit(self, backstress, d_epsilon, sign):
        """
        Update backstress using implicit method (Backward Euler) for uniaxial case.
        
        Implicit equation to solve for α_new:
        α_new = α_old + C * dp * sign - γ * (μ + H_gi * (1-μ)) * dp * α_new
        
        This is non-linear because H_gi depends on |α_new|.
        We solve for the magnitude |α_new| first, then recover the sign.
        No Macaulay bracket: all directions contribute.
        """
        new_backstress = backstress.copy()
        dp = d_epsilon
        
        for i in range(self.n_components):
            alpha_old = backstress[i]
            C = self.C[i]
            gam = self.gamma[i]
            mu = self.mu[i]
            
            # RHS of equation: RHS = alpha_old + C * dp * sign
            RHS = alpha_old + C * dp * sign
            RHS_mag = abs(RHS)
            RHS_sign = np.sign(RHS) if abs(RHS) > 1e-16 else (1.0 if sign > 0 else -1.0)
            
            # If no recovery (gamma = 0) or no step, solution is trivial
            if gam == 0 or dp < 1e-16:
                new_backstress[i] = RHS
                continue
            
            # Ren implicit update - uniaxial derivation from multi-axial theory
            # No Macaulay bracket: all directions contribute
            # Define parameters
            ri = C / gam if gam > 0 else 1e10
            
            # Solve for magnitude y = |α_new|
            # Implicit equation: α_new = α_old + C * dp * sign - γ * (μ + H_gi*(1-μ)) * dp * α_new
            # Rearranging: α_new + γ * (μ + H_gi*(1-μ)) * dp * α_new = α_old + C * dp * sign
            # In terms of magnitude: y * (1 + recovery_coeff * dp) = |RHS|
            # where recovery_coeff = γ * (μ + H_gi*(1-μ)) (no Macaulay bracket)
            def residual_signed(x):
                # Residual: x * (1 + recovery_coeff * dp) - RHS
                abs_x = abs(x)
                gi = abs_x**2 - ri**2
                H_gi = self.heaviside_sigmoidal(gi)
                recovery_coeff = gam * (mu + H_gi * (1.0 - mu))
                return x * (1.0 + recovery_coeff * dp) - RHS
            
            # Initial guess: explicit update
            x_explicit = alpha_old + C * dp * sign
            if abs(alpha_old) > 1e-14:
                ai_bar = abs(alpha_old)
                gi = ai_bar**2 - ri**2
                H_gi = self.heaviside_sigmoidal(gi)
                recovery_term = gam * (mu + H_gi * (1.0 - mu)) * dp * alpha_old
                x_explicit -= recovery_term
            
            # Determine bracket
            if RHS > 0:
                lower = 0.0
                upper = max(RHS * 1.5, x_explicit * 1.5) if x_explicit > 0 else RHS * 1.5
            else:
                lower = min(RHS * 1.5, x_explicit * 1.5) if x_explicit < 0 else RHS * 1.5
                upper = 0.0
            
            # Solve with brentq
            try:
                from scipy.optimize import brentq, newton
                res_lower = residual_signed(lower)
                res_upper = residual_signed(upper)
                
                if res_lower * res_upper > 0:
                    try:
                        x = newton(residual_signed, x_explicit, tol=1e-12, maxiter=50)
                    except:
                        x = x_explicit
                else:
                    x = brentq(residual_signed, lower, upper, xtol=1e-12, maxiter=100)
            except Exception as e:
                # Fallback: use simple iteration or explicit update
                try:
                    low, high = lower, upper
                    for _ in range(50):
                        mid = 0.5 * (low + high)
                        res_mid = residual_mag(mid)
                        if abs(res_mid) < 1e-12:
                            x = mid * RHS_sign
                            break
                        elif res_mid > 0:
                            high = mid
                        else:
                            low = mid
                    else:
                        # If bisection didn't converge, use explicit update as fallback
                        delta_alpha = C * dp * sign
                        if abs(alpha_old) > 1e-14:
                            # Use explicit Ren update formula (uniaxial derivation)
                            ai_bar = abs(alpha_old)  # Uniaxial: ᾱ = |α|
                            ri_val = C / gam if gam > 0 else 1e10
                            gi = ai_bar**2 - ri_val**2
                            H_gi = self.heaviside_sigmoidal(gi)
                            recovery_term = gam * (mu + H_gi * (1.0 - mu)) * dp * alpha_old
                            delta_alpha -= recovery_term
                        x = alpha_old + delta_alpha
                except Exception:
                    # Final fallback: explicit update (uniaxial derivation)
                    delta_alpha = C * dp * sign
                    if abs(alpha_old) > 1e-14:
                        # Use explicit Ren update formula (uniaxial derivation)
                        ai_bar = abs(alpha_old)  # Uniaxial: ᾱ = |α|
                        ri_val = C / gam if gam > 0 else 1e10
                        gi = ai_bar**2 - ri_val**2
                        H_gi = self.heaviside_sigmoidal(gi)
                        recovery_term = gam * (mu + H_gi * (1.0 - mu)) * dp * alpha_old
                        delta_alpha -= recovery_term
                    x = alpha_old + delta_alpha
            
            new_backstress[i] = x
            
        return new_backstress

    def update_state_vars(self, state_vars, d_epsilon, sign):
        """
        Update isotropic hardening state variables (R and p) using separate isotropic hardening model.
        """
        # Get current values from the solver state
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        
        # d_epsilon is the increment of accumulated plastic strain (dp)
        dp = d_epsilon
        
        # Update isotropic hardening using separate model (explicit method)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        
        # Update state variables
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        
        return state_vars

class RenModelMultiAxial(BaseMaterialModel):
    """
    Ren kinematic hardening model for multi-axial stress states (explicit version).
    Structure identical to AKO, only backstress and plastic modulus formulas differ (no Macaulay bracket).
    Uses sigmoidal approximation for Heaviside function: H(gi) = 1 / (1 + exp(-k * gi))
    
    Parameters:
    - k: Steepness parameter for sigmoidal Heaviside function (larger k = sharper transition)
    """
    def __init__(self, isotropic_model, C, gamma, mu, E, nu=0.3, k=1000.0):
        super().__init__(C=C, gamma=gamma, mu=mu)
        self.C = np.array(C)
        self.gamma = np.array(gamma)
        self.mu = np.array(mu)
        self.k = float(k)  # Steepness parameter for sigmoidal Heaviside function
        self.n_components = len(C)
        self.isotropic_model = isotropic_model
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))
    
    def heaviside_sigmoidal(self, gi):
        """
        Compute sigmoidal approximation of Heaviside function.
        
        Uses logistic function: H(gi) = 1 / (1 + exp(-k * gi))
        
        Args:
            gi: Input value (typically gi = ᾱ_i² - r_i²)
        
        Returns:
            Smooth approximation of Heaviside step function
            - When gi >> 0: H ≈ 1
            - When gi << 0: H ≈ 0
            - When gi = 0: H = 0.5
            - As k → ∞: approaches true Heaviside step function
        """
        # Logistic function: 1 / (1 + exp(-k * gi))
        return 1.0 / (1.0 + np.exp(-self.k * gi))

    def update_backstress(self, backstress, d_epsilon, flow_direction, state_vars):
        """
        Update backstress using explicit Ren evolution law (multi-axial).
        No Macaulay bracket: all plastic directions contribute to recovery.
        """
        new_backstress = [np.zeros((3, 3)) for _ in backstress]
        d_epsilon_p = np.sqrt(3. / 2.) * d_epsilon * flow_direction
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            n_dot_ai = np.sum(flow_direction * ai)
            ai_bar = np.sqrt(3.0/2.0) * ai_norm
            ri = self.C[i]/self.gamma[i]
            gi = ai_bar**2 - ri**2
            H_gi = self.heaviside_sigmoidal(gi)
            n_dot_ai_unit = n_dot_ai / ai_bar if ai_norm > 1e-14 else 0.0
            # Ren model: no Macaulay, all directions contribute
            delta_alpha = (2.0 / 3.0) * self.C[i] * d_epsilon_p \
                - self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * d_epsilon * ai
            new_backstress[i] = ai + delta_alpha
        return new_backstress

    def compute_plastic_modulus(self, backstress, state_vars):
        """
        Compute the Ren plastic modulus for multi-axial case.
        No Macaulay bracket: all plastic directions contribute.
        """
        flow_direction = state_vars.get('flow_direction', np.zeros((3, 3)))
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            ai_norm = np.linalg.norm(ai)
            n_dot_ai = np.sum(flow_direction * ai)
            ai_bar = np.sqrt(3.0/2.0) * ai_norm
            ri = self.C[i]/self.gamma[i]
            gi = ai_bar**2 - ri**2
            H_gi = self.heaviside_sigmoidal(gi)
            n_dot_ai_unit = n_dot_ai / ai_bar if ai_norm > 1e-14 else 0.0
            # Ren model: no Macaulay, all directions contribute
            h_kin +=  self.C[i] - np.sqrt(3/2)*self.gamma[i] * (self.mu[i] + H_gi * (1.0 - self.mu[i])) * ai_bar * n_dot_ai_unit
        h_iso = self.isotropic_model.compute_hardening_modulus(R, p)
        return h_kin + h_iso

    # All other methods (yield_function, plastic_multiplier, etc.) are identical to AKO
    def yield_function(self, trial_stress, total_backstress, sigma_y):
        eff_stress = trial_stress - total_backstress
        s = deviatoric(eff_stress)
        J2 = 0.5 * np.sum(s * s)
        f = np.sqrt(3 * J2) - sigma_y
        return f, s

    def plastic_multiplier(self, f, E, backstress, state_vars):
        h_total = self.compute_plastic_modulus(backstress, state_vars)
        denominator = 3 * self.G + h_total
        if denominator <= 1e-12:
            d_epsilon = 0.0
        else:
            d_epsilon = f / denominator
        if d_epsilon < 0:
            d_epsilon = 0.0
        return d_epsilon

    def update_state_vars(self, state_vars, dp, sign=None):
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)
        R_new, p_new = self.isotropic_model.update_explicit(R, p, dp)
        state_vars['R'] = R_new
        state_vars['p'] = p_new
        return state_vars

    def update_state_vars_implicit(self, R_old, p_old, dp):
        R_new, p_new = self.isotropic_model.update_implicit(R_old, p_old, dp)
        return R_new, p_new 