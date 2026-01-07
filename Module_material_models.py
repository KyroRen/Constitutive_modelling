#!/usr/bin/env python3
"""
Material models for plasticity computation.
This module contains different material constitutive models for plasticity analysis.
"""

import numpy as np
import ast
import os
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
        if 'ChabocheT' in model_name:
            model = ChabocheTModelUniAxial(isotropic_model=isotropic_model, **final_params)
        elif 'Chaboche' in model_name:
            model = ChabocheModelUniAxial(isotropic_model=isotropic_model, **final_params)
        elif 'OWII' in model_name or 'OhnoWang' in model_name:
            model = OhnoWangIIModelUniAxial(isotropic_model=isotropic_model, **final_params)
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
        sign = state_vars.get('sign', 1.0)
        R = state_vars.get('R', 0.0)
        p = state_vars.get('p', 0.0)

        # Standard Chaboche for components 0..n-2, modified for the 4th (index 3)
        h_kin = 0.0
        for i in range(self.n_components):
            ai = backstress[i]
            if i == 3:  # thresholded component
                # Effective gamma reduced by (1 - al / J(a4))_+
                Ja = abs(ai)
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
        new_backstress = backstress.copy()
        for i in range(self.n_components):
            if i == 3:
                ai = backstress[i]
                Ja = abs(ai)
                factor = max(0.0, 1.0 - (self.al / Ja) if Ja > 1e-16 else 0.0)
                delta_alpha = self.C[i] * d_epsilon * sign - self.gamma[i] * factor * ai * d_epsilon
            else:
                delta_alpha = self.C[i] * d_epsilon * sign - self.gamma[i] * backstress[i] * d_epsilon
            new_backstress[i] += delta_alpha
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
        
        # # For numerical consistency with uniaxial model, check if this is essentially uniaxial
        # # If stress tensor is nearly uniaxial, use the simpler uniaxial formula
        # if (abs(eff_stress[1,1]) < 1e-10 and abs(eff_stress[2,2]) < 1e-10 and 
        #     abs(eff_stress[0,1]) < 1e-10 and abs(eff_stress[0,2]) < 1e-10 and abs(eff_stress[1,2]) < 1e-10):
        #     # This is essentially uniaxial stress state
        #     f_uniaxial = abs(eff_stress[0,0]) - sigma_y
        #     # Use the uniaxial result for better numerical consistency
        #     if abs(f - f_uniaxial) > 1e-12:
        #         f = f_uniaxial
                
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
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_norm / denom) ** self.m[i] if denom != 0 else 0.0
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
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_norm / denom) ** self.m[i] if denom != 0 else 0.0
            delta_alpha = (2.0 / 3.0) * self.C[i] * d_epsilon_p - self.gamma[i] * pow_term * d_epsilon * ai
            new_backstress[i] = ai + delta_alpha
        return new_backstress

    def update_backstress_implicit(self, backstress_tensors_old, dp, flow_direction):
        """
        Implicit update for Ohno-Wang II backstress (multi-axial, fully implicit step).
        Uses a simple backward Euler step (can be improved with Newton if needed).
        """
        new_backstress = [np.zeros((3, 3)) for _ in backstress_tensors_old]
        d_epsilon_p = np.sqrt(3. / 2.) * dp * flow_direction
        for i in range(self.n_components):
            ai_old = backstress_tensors_old[i]
            ai_norm = np.linalg.norm(ai_old)
            denom = self.C[i] / self.gamma[i] if self.gamma[i] != 0 else 1.0
            pow_term = (ai_norm / denom) ** self.m[i] if denom != 0 else 0.0
            # Backward Euler: a_i_new = (a_i_old + (2/3)*C_i*dε_p) / (1 + γ_i*pow_term*dp)
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