#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material solver module for plasticity computation.
Cross-platform compatible for Windows, macOS, and Linux.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, root
from .material_models import *


def generate_cyclic_path_flexible(amp_pos, amp_neg, n_cycles, n_points, input_type='UA'):
    """
    Flexible cyclic path generation function that handles both uniaxial and multiaxial cases.
    
    Parameters:
    -----------
    amp_pos : float or array-like
        Maximum amplitude value(s). Can be:
        - Single float: for uniaxial loading
        - 1D array: for uniaxial loading with multiple components
        - 3x3 tensor: for multiaxial loading
    amp_neg : float or array-like
        Minimum amplitude value(s). Same format as amp_pos.
    n_cycles : int
        Number of cycles to generate
    n_points : int
        Number of points per cycle segment
    input_type : str, optional
        'UA' for uniaxial pattern, 'MA' for multiaxial pattern (default: 'UA')
    
    Returns:
    --------
    path : ndarray
        Generated cyclic path. Shape depends on input_type:
        - 'UA': 1D array for uniaxial loading
        - 'MA': 3D array (n_points_total, 3, 3) for multiaxial loading
    cycle_ids : ndarray
        Cycle index for each point
    sequence_ids : ndarray
        Sequence index for each point
    
    Notes:
    ------
    - For 3D tensor input with input_type='UA', extracts [0,0] component with warning
    - For 1D input with input_type='MA', creates uniaxial stress state (σ11 only)
    """
    
    # Convert inputs to numpy arrays for consistent handling
    amp_pos = np.asarray(amp_pos)
    amp_neg = np.asarray(amp_neg)
    
    # Determine input dimensionality
    if amp_pos.ndim == 0:  # Scalar input
        is_1d = True
        amp_pos_val = float(amp_pos)
        amp_neg_val = float(amp_neg)
    elif amp_pos.ndim == 1:  # 1D array input
        is_1d = True
        if len(amp_pos) == 1:
            amp_pos_val = float(amp_pos[0])
            amp_neg_val = float(amp_neg[0])
        else:
            # Multiple components - use first one for uniaxial pattern
            amp_pos_val = float(amp_pos[0])
            amp_neg_val = float(amp_neg[0])
    elif amp_pos.ndim == 2 and amp_pos.shape == (3, 3):  # 3x3 tensor input
        is_1d = False
        if input_type == 'UA':
            # Extract [0,0] component with warning
            print("Warning: 3D tensor input with input_type='UA' - extracting [0,0] component only")
            amp_pos_val = float(amp_pos[0, 0])
            amp_neg_val = float(amp_neg[0, 0])
        else:  # MA case
            amp_pos_val = amp_pos.copy()
            amp_neg_val = amp_neg.copy()
    else:
        raise ValueError(f"Unsupported input shape: {amp_pos.shape}. Expected scalar, 1D array, or 3x3 tensor.")
    
    # Generate the basic cyclic pattern (1D)
    path_1d = []
    cycle_ids = []
    sequence_ids = []
    seq = 0
    
    # First cycle: loading (0->+), unloading (+->-), loading (- -> +)
    seg = np.linspace(0, amp_pos_val, n_points)
    path_1d.extend(seg)
    cycle_ids.extend([0]*n_points)
    sequence_ids.extend([seq]*n_points)
    seq += 1
    
    seg = np.linspace(amp_pos_val, amp_neg_val, n_points)
    path_1d.extend(seg)
    cycle_ids.extend([0]*n_points)
    sequence_ids.extend([seq]*n_points)
    seq += 1
    
    seg = np.linspace(amp_neg_val, amp_pos_val, n_points)
    path_1d.extend(seg)
    cycle_ids.extend([0]*n_points)
    sequence_ids.extend([seq]*n_points)
    seq += 1
    
    # Subsequent cycles: unloading (+->-), loading (- -> +)
    for c in range(1, n_cycles):
        seg = np.linspace(amp_pos_val, amp_neg_val, n_points)
        path_1d.extend(seg)
        cycle_ids.extend([c]*n_points)
        sequence_ids.extend([seq]*n_points)
        seq += 1
        
        seg = np.linspace(amp_neg_val, amp_pos_val, n_points)
        path_1d.extend(seg)
        cycle_ids.extend([c]*n_points)
        sequence_ids.extend([seq]*n_points)
        seq += 1
    
    path_1d = np.array(path_1d)
    cycle_ids = np.array(cycle_ids)
    sequence_ids = np.array(sequence_ids)
    
    # Convert to appropriate output format based on input_type
    if input_type == 'UA':
        # Return 1D array for uniaxial loading
        return path_1d, cycle_ids, sequence_ids
    
    elif input_type == 'MA':
        # Convert to 3D tensor format for multiaxial loading
        n_total_points = len(path_1d)
        path_3d = np.zeros((n_total_points, 3, 3))
        
        if is_1d:
            # 1D input with MA output: create uniaxial stress state (σ11 only)
            path_3d[:, 0, 0] = path_1d
        else:
            # 3D tensor input with MA output: scale the tensor pattern
            for i in range(n_total_points):
                # Linear interpolation between amp_neg and amp_pos tensors
                # Extract scalar values for interpolation
                amp_pos_scalar = amp_pos_val[0, 0] if isinstance(amp_pos_val, np.ndarray) else amp_pos_val
                amp_neg_scalar = amp_neg_val[0, 0] if isinstance(amp_neg_val, np.ndarray) else amp_neg_val
                
                if amp_pos_scalar != amp_neg_scalar:
                    alpha = (path_1d[i] - amp_neg_scalar) / (amp_pos_scalar - amp_neg_scalar)
                else:
                    alpha = 0.5
                
                path_3d[i] = amp_neg_val + alpha * (amp_pos_val - amp_neg_val)
        
        return path_3d, cycle_ids, sequence_ids
    
    else:
        raise ValueError(f"Invalid input_type: {input_type}. Must be 'UA' or 'MA'.")


class UnifiedMaterialSolverMultiAxial:
    """
    General stress-strain solver for multi-axial models with explicit/implicit method switching.
    Stand-alone solver that doesn't inherit from any base class.
    """
    def __init__(self, E, nu, yield_stress, material_model, method='fully_implicit', precision='standard'):
        """
        Initialize the multi-axial solver.
        
        Parameters:
        - E: Young's modulus
        - nu: Poisson's ratio
        - yield_stress: Initial yield stress
        - material_model: Material model object
        - method: 'explicit', 'implicit_explicit', or 'fully_implicit'
        - precision: 'standard', 'high', or 'scientific'
        """
        self.E = E
        self.nu = nu
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self.sigma_y = yield_stress
        self.material_model = material_model
        self.n_components = getattr(material_model, 'n_components', 1)
        self.method = method
        self.precision = precision
        
        # Flag to control when plastic strain history should be stored
        self._store_plastic_strain_history = True
        
        # Set tolerance based on precision level
        self._setup_method_parameters()
        self.reset_state()

    def _setup_method_parameters(self):
        """
        Set up method-specific parameters for tolerance and precision.
        """
        precision_settings = {
            'standard': 1e-8,
            'high': 1e-10,
            'scientific': 1e-12
        }
        
        self._yield_tolerance = precision_settings.get(self.precision, 1e-8)
        self._dp_tolerance = precision_settings.get(self.precision, 1e-8)
        
        # Prevent too tight tolerances that can cause oscillations
        if self._yield_tolerance < 1e-10:
            self._yield_tolerance = 1e-10
        if self._dp_tolerance < 1e-10:
            self._dp_tolerance = 1e-10
            
        # **NEW: Unified stress control tolerances**
        self._stress_control_tolerance = self._yield_tolerance
        self._stress_control_max_iter = 50
        self._stress_control_methods = ['hybr', 'lm', 'broyden1']
        self._stress_control_tolerance_levels = [
            self._stress_control_tolerance,
            self._stress_control_tolerance * 10,
            self._stress_control_tolerance * 100
        ]

    def set_method(self, method='fully_implicit', precision=None):
        """
        Switch between explicit, implicit_explicit (IMEX), and fully_implicit methods.
        """
        old_method = self.method
        self.method = method
        
        if precision is not None:
            self.precision = precision
            
        self._setup_method_parameters()
        
        # Print method switching information (keep this as solver indicator)
        print(f"Switched from {old_method} to {method} method with {self.precision} precision")
        # print(f"Yield tolerance: {self._yield_tolerance}")

    def reset_state(self):
        """Reset all state variables to initial conditions."""
        self.plastic_strain = np.zeros((3, 3))
        self.backstress = [np.zeros((3, 3)) for _ in range(self.n_components)]
        self.current_stress = np.zeros((3, 3))
        self.total_strain = np.zeros((3, 3))
        self.R = 0.0
        self.p = 0.0
        self.dp_history = []
        self.plastic_strain_history = []
        self._last_f_value = None
        
        # Reset plastic strain smoothing state
        # self._previous_plastic_strain = None # Removed smoothing state

    def compute_stress(self, total_strain_tensor, n_substeps=1):
        """
        Compute stress using the selected method.
        - explicit: full explicit integration
        - implicit_explicit: implicit yield (solve for dp), explicit backstress/state updates (IMEX)
        - fully_implicit: coupled residual for dp and backstress
        """
        if self.method == 'explicit':
            return self._compute_stress_explicit(total_strain_tensor, n_substeps)
        elif self.method == 'implicit_explicit':
            return self._compute_stress_implicit_explicit(total_strain_tensor, n_substeps)
        else:  # fully_implicit
            return self._compute_stress_fully_implicit(total_strain_tensor, n_substeps)
    
    def _compute_stress_explicit(self, total_strain_tensor, n_substeps=1):
        """
        Explicit stress computation with sub-stepping for robustness.
        """
        # Check if we're crossing the yield point
        elastic_strain_trial = total_strain_tensor - self.plastic_strain
        trace_e_trial = np.trace(elastic_strain_trial)
        trial_stress = self.lmbda * trace_e_trial * np.identity(3) + 2 * self.mu * elastic_strain_trial
        
        total_backstress = np.sum(self.backstress, axis=0)
        current_yield_stress = self.sigma_y + self.R
        f_trial, _ = self.material_model.yield_function(trial_stress, total_backstress, current_yield_stress)
        
        # If we're crossing from elastic to plastic, add an intermediate step at exact yield
        if f_trial > 1e-8 and hasattr(self, '_last_stress_tensor'):
            # Check if previous step was elastic
            prev_f, _ = self.material_model.yield_function(self._last_stress_tensor, total_backstress, current_yield_stress)
            if prev_f <= 1e-8:
                # Find strain that gives exact yield
                exact_yield_strain_tensor = self._find_exact_yield_strain(total_strain_tensor)
                if exact_yield_strain_tensor is not None:
                    # Compute stress at exact yield point first
                    yield_stress = self._compute_stress_internal_explicit(exact_yield_strain_tensor)
                    self._yield_point_stress = yield_stress
        
        # Now compute the actual stress for the target strain
        result = self._compute_stress_internal_explicit(total_strain_tensor)
        self._last_stress_tensor = result.copy()
        return result
    
    def _compute_stress_fully_implicit(self, total_strain_tensor, n_substeps=1):
        """
        Fully-implicit stress computation with adaptive yield point detection.
        """
        # Check if we're crossing the yield point
        elastic_strain_trial = total_strain_tensor - self.plastic_strain
        trace_e_trial = np.trace(elastic_strain_trial)
        trial_stress = self.lmbda * trace_e_trial * np.identity(3) + 2 * self.mu * elastic_strain_trial
        
        total_backstress = np.sum(self.backstress, axis=0)
        current_yield_stress = self.sigma_y + self.R
        f_trial, _ = self.material_model.yield_function(trial_stress, total_backstress, current_yield_stress)
        
        # If we're crossing from elastic to plastic, add an intermediate step at exact yield
        if f_trial > 1e-8 and hasattr(self, '_last_stress_tensor'):
            # Check if previous step was elastic
            prev_f, _ = self.material_model.yield_function(self._last_stress_tensor, total_backstress, current_yield_stress)
            if prev_f <= 1e-8:
                # Find strain that gives exact yield
                exact_yield_strain_tensor = self._find_exact_yield_strain(total_strain_tensor)
                if exact_yield_strain_tensor is not None:
                    # Compute stress at exact yield point first
                    yield_stress = self._compute_stress_internal_fully_implicit(exact_yield_strain_tensor, n_substeps)
                    self._yield_point_stress = yield_stress
        
        # Now compute the actual stress for the target strain
        result = self._compute_stress_internal_fully_implicit(total_strain_tensor, n_substeps)
        self._last_stress_tensor = result.copy()
        return result

    def _compute_stress_implicit_explicit(self, total_strain_tensor, n_substeps=1):
        """
        Implicit-explicit (IMEX) stress computation: implicit yield consistency,
        explicit backstress and isotropic hardening updates.
        """
        # Check if we're crossing the yield point
        elastic_strain_trial = total_strain_tensor - self.plastic_strain
        trace_e_trial = np.trace(elastic_strain_trial)
        trial_stress = self.lmbda * trace_e_trial * np.identity(3) + 2 * self.mu * elastic_strain_trial
        
        total_backstress = np.sum(self.backstress, axis=0)
        current_yield_stress = self.sigma_y + self.R
        f_trial, _ = self.material_model.yield_function(trial_stress, total_backstress, current_yield_stress)
        
        # If we're crossing from elastic to plastic, add an intermediate step at exact yield
        if f_trial > 1e-8 and hasattr(self, '_last_stress_tensor'):
            prev_f, _ = self.material_model.yield_function(self._last_stress_tensor, total_backstress, current_yield_stress)
            if prev_f <= 1e-8:
                exact_yield_strain_tensor = self._find_exact_yield_strain(total_strain_tensor)
                if exact_yield_strain_tensor is not None:
                    yield_stress = self._compute_stress_internal_implicit_explicit(exact_yield_strain_tensor, n_substeps)
                    self._yield_point_stress = yield_stress
        
        result = self._compute_stress_internal_implicit_explicit(total_strain_tensor, n_substeps)
        self._last_stress_tensor = result.copy()
        return result
    
    def _find_exact_yield_strain(self, target_strain_tensor):
        """
        Find the strain tensor that gives exact yield stress using bisection.
        """
        current_strain = self.total_strain.copy()
        target_strain = target_strain_tensor.copy()
        
        # Use bisection to find exact yield point
        for _ in range(20):  # Max 20 iterations
            mid_strain = (current_strain + target_strain) / 2.0
            
            # Test stress at mid point
            elastic_strain_mid = mid_strain - self.plastic_strain
            trace_e_mid = np.trace(elastic_strain_mid)
            trial_stress_mid = self.lmbda * trace_e_mid * np.identity(3) + 2 * self.mu * elastic_strain_mid
            
            total_backstress = np.sum(self.backstress, axis=0)
            f_mid, _ = self.material_model.yield_function(trial_stress_mid, total_backstress, self.sigma_y + self.R)
            
            if abs(f_mid) < 1e-10:  # Found exact yield point
                return mid_strain
            elif f_mid < 0:  # Still elastic
                current_strain = mid_strain
            else:  # Already plastic
                target_strain = mid_strain
        
        return (current_strain + target_strain) / 2.0  # Return best approximation
    
    def _compute_stress_internal_explicit(self, total_strain_tensor, n_substeps=1):
        """
        Internal explicit stress computation method with sub-stepping.
        """
        d_total_strain = total_strain_tensor - self.total_strain

        # No need to proceed if strain increment is negligible
        if mag(d_total_strain) < 1e-12:
            # Still need to store history for plotting consistency
            self.dp_history.append(0.0)  # No plastic activity
            if self._store_plastic_strain_history:
                self.plastic_strain_history.append(self.plastic_strain.copy())
            return self.current_stress

        # Determine number of sub-steps based on strain increment size
        strain_increment_size = mag(d_total_strain)
        if strain_increment_size > 0.001:  # 0.1% strain
            n_substeps = max(int(strain_increment_size / 0.001), 2)
        else:
            n_substeps = 1

        # Limit maximum sub-steps to prevent excessive computation
        n_substeps = min(n_substeps, 20)
        n_substeps = 20
        # Sub-stepping
        d_strain_sub = d_total_strain / n_substeps
        dp_total_step = 0.0

        # Track previous yield function value for unloading detection
        f_prev = None
        if hasattr(self, '_last_f_value'):
            f_prev = self._last_f_value
        
        for i in range(n_substeps):
            sub_step_target_strain = self.total_strain + d_strain_sub
            elastic_strain_sub_trial = sub_step_target_strain - self.plastic_strain
            trace_e_sub_trial = np.trace(elastic_strain_sub_trial)
            trial_stress_sub = self.lmbda * trace_e_sub_trial * np.identity(3) + 2 * self.mu * elastic_strain_sub_trial
            total_backstress_sub = np.sum(self.backstress, axis=0)
            current_yield_stress = self.sigma_y + self.R
            f_sub, s_trial_sub = self.material_model.yield_function(trial_stress_sub, total_backstress_sub, current_yield_stress)

            # Enhanced yield check for better elastic-plastic transition detection
            effective_tolerance = self._yield_tolerance
            
            # Detect unloading: if yield function is decreasing (moving towards elastic region)
            is_unloading = False
            if f_prev is not None and f_sub < f_prev and f_sub < self._yield_tolerance * 5:
                is_unloading = True
                # Use stricter tolerance for unloading to ensure proper elastic transition
                effective_tolerance = self._yield_tolerance * 0.1
                
            if f_sub > effective_tolerance:
                # Plastic step
                s_trial_norm = mag(s_trial_sub)
                flow_direction = s_trial_sub / s_trial_norm if s_trial_norm > 1e-12 else np.zeros((3, 3))
                
                # Prepare state variables for material model
                state_vars = {'flow_direction': flow_direction, 'R': self.R, 'p': self.p}
                
                # Use explicit plastic multiplier calculation
                d_epsilon = self.material_model.plastic_multiplier(f_sub, 2*self.mu, self.backstress, state_vars)
                
                # Limit plastic multiplier to prevent excessive increments
                d_epsilon = min(d_epsilon, 0.001)  # Limit to 0.1% strain per sub-step
                
                # Update backstress using explicit method
                self.backstress = self.material_model.update_backstress(self.backstress, d_epsilon, flow_direction, state_vars)
                
                # Update state variables using explicit method (IMEX: explicit for all hardening variables)
                updated_state_vars = self.material_model.update_state_vars(state_vars, d_epsilon)
                self.R = updated_state_vars.get('R', self.R)
                self.p = updated_state_vars.get('p', self.p)
                
                # Update plastic strain and stress
                d_epsilon_p = np.sqrt(3./2.) * d_epsilon * flow_direction
                self.plastic_strain += d_epsilon_p
                self.current_stress = trial_stress_sub - 2 * self.mu * d_epsilon_p
                
                dp_total_step += d_epsilon
            else:
                # Elastic step
                self.current_stress = trial_stress_sub
                
            self.total_strain = sub_step_target_strain
            
            # Update previous yield function value for next substep
            f_prev = f_sub

        # Store final yield function value for next call
        self._last_f_value = f_sub
        self.dp_history.append(dp_total_step)
        
        # Store plastic strain history for plotting (only if flag is True)
        if self._store_plastic_strain_history:
            self.plastic_strain_history.append(self.plastic_strain.copy())
        return self.current_stress

    def _compute_stress_internal_implicit_explicit(self, total_strain_tensor, n_substeps=1):
        """
        Internal IMEX stress computation: implicit yield consistency (solve for dp),
        explicit backstress and isotropic hardening updates.
        """
        d_total_strain = total_strain_tensor - self.total_strain

        if mag(d_total_strain) < 1e-12:
            self.dp_history.append(0.0)
            if self._store_plastic_strain_history:
                self.plastic_strain_history.append(self.plastic_strain.copy())
            return self.current_stress

        d_strain_sub = d_total_strain / n_substeps
        dp_total_step = 0.0
        f_prev = getattr(self, '_last_f_value', None)

        for _ in range(n_substeps):
            sub_step_target_strain = self.total_strain + d_strain_sub
            elastic_strain_sub_trial = sub_step_target_strain - self.plastic_strain
            trace_e_sub_trial = np.trace(elastic_strain_sub_trial)
            trial_stress_sub = self.lmbda * trace_e_sub_trial * np.identity(3) + 2 * self.mu * elastic_strain_sub_trial
            total_backstress_sub = np.sum(self.backstress, axis=0)
            f_sub, s_trial_sub = self.material_model.yield_function(trial_stress_sub, total_backstress_sub, self.sigma_y + self.R)

            effective_tolerance = self._yield_tolerance
            is_unloading = False
            if f_prev is not None and f_sub < f_prev and f_sub < self._yield_tolerance * 5:
                is_unloading = True
                effective_tolerance = self._yield_tolerance * 0.1

            if f_sub > effective_tolerance:
                s_trial_norm = mag(s_trial_sub)
                flow_direction = s_trial_sub / s_trial_norm if s_trial_norm > 1e-12 else np.zeros((3, 3))
                state_vars = {'flow_direction': flow_direction, 'R': self.R, 'p': self.p}

                def residual_yield(dp_guess):
                    """Residual: yield function f evaluated at state after explicit update with dp."""
                    dp_reg = max(dp_guess, 0.0)
                    # Explicit backstress update
                    alpha_new = self.material_model.update_backstress(self.backstress, dp_reg, flow_direction, state_vars)
                    # Explicit state vars update
                    updated_sv = self.material_model.update_state_vars(state_vars.copy(), dp_reg)
                    R_new = updated_sv.get('R', self.R)
                    # Compute stress and yield
                    d_epsilon_p = np.sqrt(3.0 / 2.0) * dp_reg * flow_direction
                    stress_new = trial_stress_sub - 2 * self.mu * d_epsilon_p
                    total_backstress_new = np.sum(alpha_new, axis=0)
                    f_val, _ = self.material_model.yield_function(stress_new, total_backstress_new, self.sigma_y + R_new)
                    return f_val

                # Initial guess for dp
                h_plastic = self.material_model.compute_plastic_modulus(self.backstress, state_vars)
                dp_init = f_sub / (3 * self.mu + h_plastic) if (3 * self.mu + h_plastic) > 1e-6 else 0.0
                dp_init = min(dp_init, 0.001)

                try:
                    dp_solution = newton(residual_yield, dp_init, tol=self._dp_tolerance, maxiter=50)
                    final_dp = max(dp_solution, 0.0)
                except (RuntimeError, ValueError):
                    # Fallback: use explicit plastic multiplier
                    final_dp = self.material_model.plastic_multiplier(f_sub, 2 * self.mu, self.backstress, state_vars)
                    final_dp = min(final_dp, 0.001)

                # Apply explicit updates with the solved dp
                self.backstress = self.material_model.update_backstress(self.backstress, final_dp, flow_direction, state_vars)
                updated_sv = self.material_model.update_state_vars(state_vars, final_dp)
                self.R = updated_sv.get('R', self.R)
                self.p = updated_sv.get('p', self.p)

                d_epsilon_p = np.sqrt(3.0 / 2.0) * final_dp * flow_direction
                self.plastic_strain += d_epsilon_p
                self.current_stress = trial_stress_sub - 2 * self.mu * d_epsilon_p
                dp_total_step += final_dp
            else:
                self.current_stress = trial_stress_sub

            self.total_strain = sub_step_target_strain
            f_prev = f_sub

        self._last_f_value = f_sub
        self.dp_history.append(dp_total_step)
        if self._store_plastic_strain_history:
            self.plastic_strain_history.append(self.plastic_strain.copy())
        return self.current_stress
    
    def _compute_stress_internal_fully_implicit(self, total_strain_tensor, n_substeps=1):
        """
        Internal stress computation with robust root-finding approach (fully-implicit).
        """
        d_total_strain = total_strain_tensor - self.total_strain

        # No need to proceed if strain increment is negligible
        if mag(d_total_strain) < 1e-12:
            # Still need to store history for plotting consistency
            self.dp_history.append(0.0)  # No plastic activity
            if self._store_plastic_strain_history:
                self.plastic_strain_history.append(self.plastic_strain.copy())
            return self.current_stress

        # Sub-stepping for robustness
        d_strain_sub = d_total_strain / n_substeps
        dp_total_step = 0.0
        
        for _ in range(n_substeps):
            sub_step_target_strain = self.total_strain + d_strain_sub
            elastic_strain_sub_trial = sub_step_target_strain - self.plastic_strain
            trace_e_sub_trial = np.trace(elastic_strain_sub_trial)
            trial_stress_sub = self.lmbda * trace_e_sub_trial * np.identity(3) + 2 * self.mu * elastic_strain_sub_trial
            total_backstress_sub_start = np.sum(self.backstress, axis=0)
            f_sub, s_trial_sub = self.material_model.yield_function(trial_stress_sub, total_backstress_sub_start, self.sigma_y + self.R)

            # Use consistent tolerance with explicit method for better consistency
            if f_sub > self._yield_tolerance:
                s_trial_norm = mag(s_trial_sub)
                flow_direction = s_trial_sub / s_trial_norm if s_trial_norm > 1e-12 else np.zeros((3,3))

                # Cache material model parameters for the residual closure
                model = self.material_model
                C_arr = model.C
                gamma_arr = model.gamma
                has_pow_term = hasattr(model, 'm')  # True for OWII
                has_ako_kang = hasattr(model, 'mu') and not has_pow_term  # Kang, AKO (mu not m)
                use_macaulay = (has_ako_kang and 'AbdelKarimOhno' in type(model).__name__)
                if has_pow_term:
                    m_arr = model.m
                if has_ako_kang:
                    mu_arr = np.atleast_1d(model.mu)

                def residual_coupled(x):
                    dp_guess = x[0]
                    alpha_new_list = [x[1 + i*9 : 1 + (i+1)*9].reshape(3, 3) for i in range(self.n_components)]

                    d_epsilon_p = np.sqrt(3./2.) * dp_guess * flow_direction

                    # --- Backstress backward Euler residuals (truly implicit) ---
                    # Chaboche:  α_new*(1 + γ*dp) - α_old - (2/3)*C*dε_p = 0
                    # OWII:     α_new*(1 + γ*(ā_new/r)^m*dp) - α_old - (2/3)*C*dε_p = 0
                    # Kang/AKO: α_new*(1 + γ*(μ+H_gi*(1-μ))*[bracket]*dp) - α_old - (2/3)*C*dε_p = 0
                    #   H_gi = 1 if ā²>r² else 0, bracket = max(0,n·ā/ā) for AKO only
                    residuals_alpha = []
                    for i in range(self.n_components):
                        ai_new = alpha_new_list[i]
                        ai_old = self.backstress[i]

                        recall_coeff = gamma_arr[i]
                        if has_pow_term:
                            ai_new_norm = np.linalg.norm(ai_new)
                            ai_new_bar = np.sqrt(3.0 / 2.0) * ai_new_norm if ai_new_norm > 1e-16 else 0.0
                            ri = C_arr[i] / gamma_arr[i] if gamma_arr[i] != 0 else 1.0
                            pow_val = (ai_new_bar / ri) ** m_arr[i] if ri > 0 else 0.0
                            recall_coeff *= pow_val
                        elif has_ako_kang:
                            ai_new_norm = np.linalg.norm(ai_new)
                            ai_new_bar = np.sqrt(3.0 / 2.0) * ai_new_norm if ai_new_norm > 1e-16 else 0.0
                            ri = C_arr[i] / gamma_arr[i] if gamma_arr[i] != 0 else 1.0
                            gi = ai_new_bar**2 - ri**2
                            H_gi = 1.0 if gi > 0 else 0.0
                            mu_i = mu_arr[i] if i < len(mu_arr) else mu_arr[0]
                            coef = mu_i + H_gi * (1.0 - mu_i)
                            bracket = 1.0
                            if use_macaulay and ai_new_bar > 1e-16:
                                n_dot_ai = np.sum(flow_direction * ai_new)
                                n_dot_ai_unit = n_dot_ai / ai_new_bar
                                bracket = max(0.0, n_dot_ai_unit)
                            recall_coeff *= coef * bracket

                        res = ai_new * (1.0 + recall_coeff * dp_guess) - ai_old - (2.0 / 3.0) * C_arr[i] * d_epsilon_p
                        residuals_alpha.append(res.flatten())
                    res_alpha = np.concatenate(residuals_alpha)

                    # --- Yield function residual ---
                    total_backstress_new = np.sum(alpha_new_list, axis=0)
                    R_new, _ = model.update_state_vars_implicit(self.R, self.p, dp_guess)
                    stress_new = trial_stress_sub - 2 * self.mu * d_epsilon_p
                    res_f, _ = model.yield_function(stress_new, total_backstress_new, self.sigma_y + R_new)
                    return np.concatenate(([res_f], res_alpha))

                x0 = np.zeros(1 + 9 * self.n_components)
                f_initial_abs = abs(f_sub)
                h_plastic = self.material_model.compute_plastic_modulus(self.backstress, {'flow_direction': flow_direction, 'R': self.R, 'p': self.p})
                x0[0] = f_initial_abs / (3 * self.mu + h_plastic) if (3 * self.mu + h_plastic) > 1e-6 else 0.0
                for i in range(self.n_components):
                    x0[1 + i*9 : 1 + (i+1)*9] = self.backstress[i].flatten()

                solution = root(residual_coupled, x0, method='hybr', tol=1e-10)

                if solution.success:
                    final_dp = max(solution.x[0], 0.0)
                    final_alphas = [solution.x[1 + i*9 : 1 + (i+1)*9].reshape(3, 3) for i in range(self.n_components)]
                    d_epsilon_p_sub = np.sqrt(3./2.) * final_dp * flow_direction
                    self.plastic_strain += d_epsilon_p_sub
                    self.backstress = final_alphas
                    self.R, self.p = self.material_model.update_state_vars_implicit(self.R, self.p, final_dp)
                    self.current_stress = trial_stress_sub - 2 * self.mu * d_epsilon_p_sub
                    dp_total_step += final_dp
                else:
                    self.current_stress = trial_stress_sub
                    dp_total_step += 1e-12
            else:
                # This sub-step is elastic, so we just update the stress.
                self.current_stress = trial_stress_sub
            
            self.total_strain = sub_step_target_strain

        self.dp_history.append(dp_total_step)
        
        # Store plastic strain history for plotting (only if flag is True)
        if self._store_plastic_strain_history:
            self.plastic_strain_history.append(self.plastic_strain.copy())
        return self.current_stress

    def compute_strain(self, target_stress_tensor, tol=None, max_iter=None):
        """
        Stress-controlled strain update using a multi-variable Newton-Raphson method.
        Enhanced with unified convergence handling for better consistency between solvers.
        """
        # **NEW: Use unified tolerance settings**
        if tol is None:
            tol = self._stress_control_tolerance
        if max_iter is None:
            max_iter = self._stress_control_max_iter
            
        def residual(delta_strain_voigt):
            backup_state = {
                'ps': self.plastic_strain.copy(), 'bs': [b.copy() for b in self.backstress],
                'ts': self.total_strain.copy(), 'r': self.R, 'p': self.p, 'dp_hist': self.dp_history.copy(),
                'plastic_strain_hist': self.plastic_strain_history.copy()
            }
            
            # Disable plastic strain history storage during Newton-Raphson iterations
            self._store_plastic_strain_history = False
            
            delta_strain_tensor = voigt_to_tensor(delta_strain_voigt)
            trial_strain = self.total_strain + delta_strain_tensor
            try:
                computed_stress_tensor = self.compute_stress(trial_strain)
            except:
                # If compute_stress fails, return a large residual
                self.plastic_strain = backup_state['ps']
                self.backstress = backup_state['bs']
                self.total_strain = backup_state['ts']
                self.R = backup_state['r']
                self.p = backup_state['p']
                self.dp_history = backup_state['dp_hist']
                self.plastic_strain_history = backup_state['plastic_strain_hist']
                return np.ones(6) * 1e6
                
            self.plastic_strain = backup_state['ps']
            self.backstress = backup_state['bs']
            self.total_strain = backup_state['ts']
            self.R = backup_state['r']
            self.p = backup_state['p']
            self.dp_history = backup_state['dp_hist']
            self.plastic_strain_history = backup_state['plastic_strain_hist']
            error_tensor = computed_stress_tensor - target_stress_tensor
            return tensor_to_voigt(error_tensor)

        delta_stress_tensor = target_stress_tensor - self.current_stress
        
        # **IMPROVED: Enhanced initial guess strategy**
        initial_guess_strain_tensor = np.zeros((3,3))
        trace_ds = np.trace(delta_stress_tensor)
        inv_E = 1./self.E
        initial_guess_strain_tensor = (inv_E * (1+self.nu)) * delta_stress_tensor - (inv_E * self.nu) * trace_ds * np.identity(3)
        
        # **NEW: Adaptive scaling based on stress magnitude**
        stress_magnitude = np.max(np.abs(target_stress_tensor))
        if stress_magnitude > 100:  # High stress level
            scaling_factor = 0.3
        elif stress_magnitude > 10:  # Medium stress level
            scaling_factor = 0.5
        else:  # Low stress level
            scaling_factor = 0.7
        initial_guess_strain_tensor *= scaling_factor
        initial_guess_voigt = tensor_to_voigt(initial_guess_strain_tensor)
        
        # **NEW: Unified multi-method approach**
        solution = None
        
        for method in self._stress_control_methods:
            for tolerance in self._stress_control_tolerance_levels:
                try:
                    solution = root(residual, initial_guess_voigt, tol=tolerance, method=method)
                    if solution.success and np.max(np.abs(solution.fun)) < tolerance * 10:
                        break
                except:
                    continue
            if solution and solution.success:
                break
        
        if not solution or not solution.success:
            # **IMPROVED: Enhanced fallback strategy**
            stress_diff = mag(target_stress_tensor - self.current_stress)
            if stress_diff < 1.0:  # Very small stress increment
                final_delta_strain_voigt = initial_guess_voigt * 0.1
                print(f"Warning: Using elastic fallback for small stress increment {stress_diff:.3f} MPa")
            else:
                # Try with smaller initial guess
                for scale in [0.1, 0.05, 0.01]:
                    try:
                        scaled_guess = initial_guess_voigt * scale
                        solution = root(residual, scaled_guess, tol=self._stress_control_tolerance_levels[-1], 
                                      method='hybr')
                        if solution.success:
                            break
                    except:
                        continue
                        
                if not solution or not solution.success:
                    raise RuntimeError(f"Convergence failed in stress control at stress level {np.max(np.abs(target_stress_tensor)):.2f}")
        else:
            final_delta_strain_voigt = solution.x
            
        final_delta_strain_tensor = voigt_to_tensor(final_delta_strain_voigt)
        final_strain = self.total_strain + final_delta_strain_tensor
        
        # Re-enable plastic strain history storage for final computation
        self._store_plastic_strain_history = True
        
        self.compute_stress(final_strain)
        self.current_stress = target_stress_tensor.copy()

        return self.total_strain

    def run_strain_controlled(self, strain_history_tensors):
        """
        Runs strain-controlled simulation, solving for transverse strains to ensure
        uniaxial stress state.
        """
        stresses = []
        final_strains = []
        nu = self.nu
        for i in range(len(strain_history_tensors)):
            target_e11 = strain_history_tensors[i, 0, 0]
            saved_state = {
                'plastic_strain': self.plastic_strain.copy(),
                'backstress': [b.copy() for b in self.backstress],
                'current_stress': self.current_stress.copy(),
                'total_strain': self.total_strain.copy(),
                'R': self.R, 'p': self.p, 'dp_history_len': len(self.dp_history),
                'plastic_strain_history_len': len(self.plastic_strain_history),
                '_last_stress_tensor': self._last_stress_tensor.copy() if hasattr(self, '_last_stress_tensor') and self._last_stress_tensor is not None else None,
                '_last_f_value': getattr(self, '_last_f_value', None),
            }
            def transverse_residual(e_t):
                # Restore state (including yield-detection variables to prevent zig-zag)
                self.plastic_strain = saved_state['plastic_strain'].copy()
                self.backstress = [b.copy() for b in saved_state['backstress']]
                self.current_stress = saved_state['current_stress'].copy()
                self.total_strain = saved_state['total_strain'].copy()
                self.R = saved_state['R']
                self.p = saved_state['p']
                self.dp_history = self.dp_history[:saved_state['dp_history_len']]
                self.plastic_strain_history = self.plastic_strain_history[:saved_state['plastic_strain_history_len']]
                if saved_state['_last_stress_tensor'] is not None:
                    self._last_stress_tensor = saved_state['_last_stress_tensor'].copy()
                self._last_f_value = saved_state['_last_f_value']
                
                # Disable plastic strain history storage during Newton-Raphson iterations
                self._store_plastic_strain_history = False
                
                trial_strain_tensor = np.array([[target_e11, 0, 0], [0, e_t, 0], [0, 0, e_t]])
                trial_stress = self.compute_stress(trial_strain_tensor)
                return trial_stress[1, 1]
            
            previous_e_t = self.total_strain[1, 1]
            try:
                correct_e_t = newton(transverse_residual, x0=previous_e_t, tol=1e-10, maxiter=20)
            except RuntimeError:
                correct_e_t = -nu * target_e11

            # Restore state for final computation (including yield-detection variables)
            self.plastic_strain = saved_state['plastic_strain']
            self.backstress = [b.copy() for b in saved_state['backstress']]
            self.current_stress = saved_state['current_stress']
            self.total_strain = saved_state['total_strain']
            self.R = saved_state['R']
            self.p = saved_state['p']
            self.dp_history = self.dp_history[:saved_state['dp_history_len']]
            self.plastic_strain_history = self.plastic_strain_history[:saved_state['plastic_strain_history_len']]
            if saved_state['_last_stress_tensor'] is not None:
                self._last_stress_tensor = saved_state['_last_stress_tensor'].copy()
            self._last_f_value = saved_state['_last_f_value']
            
            # Re-enable plastic strain history storage for final computation
            self._store_plastic_strain_history = True
            
            final_strain_tensor = np.array([[target_e11, 0, 0], [0, correct_e_t, 0], [0, 0, correct_e_t]])
            final_stress = self.compute_stress(final_strain_tensor)
            stresses.append(final_stress)
            final_strains.append(final_strain_tensor)
        
        return np.array(stresses), np.array(final_strains)

    def run_stress_controlled(self, stress_history_tensors):
        """ Stress-controlled simulation with tensor history. """
        strains = []
        stresses = []
        for stress_tensor in stress_history_tensors:
            try:
                strain_tensor = self.compute_strain(stress_tensor)
                strains.append(strain_tensor)
                stresses.append(self.current_stress.copy())
            except RuntimeError as e:
                # print(f"Simulation stopped: {str(e)}")
                break
        return np.array(strains), np.array(stresses)

    def get_method_info(self):
        """Return information about current method and settings."""
        return {
            'method': self.method,
            'precision': self.precision,
            'yield_tolerance': self._yield_tolerance,
            'available_methods': ['explicit', 'implicit_explicit', 'fully_implicit'],
            'available_precisions': ['standard', 'high', 'scientific']
        }

    def diagnose_stress_control(self, target_stress_tensor, n_points=10):
        """
        Diagnostic tool for stress control convergence issues.
        Analyzes the residual function behavior around the current state.
        """
        print("=== MULTIAXIAL STRESS CONTROL DIAGNOSTIC ===")
        print(f"Target stress: {np.max(np.abs(target_stress_tensor)):.3f} MPa")
        print(f"Current stress: {np.max(np.abs(self.current_stress)):.3f} MPa")
        print(f"Stress difference: {np.max(np.abs(target_stress_tensor - self.current_stress)):.3f} MPa")
        
        # Test initial guess quality
        delta_stress_tensor = target_stress_tensor - self.current_stress
        initial_guess_strain_tensor = np.zeros((3,3))
        trace_ds = np.trace(delta_stress_tensor)
        inv_E = 1./self.E
        initial_guess_strain_tensor = (inv_E * (1+self.nu)) * delta_stress_tensor - (inv_E * self.nu) * trace_ds * np.identity(3)
        
        print(f"Initial guess strain: {np.max(np.abs(initial_guess_strain_tensor)):.6f}")
        
        # Test residual at initial guess
        initial_guess_voigt = tensor_to_voigt(initial_guess_strain_tensor)
        try:
            backup_state = {
                'ps': self.plastic_strain.copy(), 'bs': [b.copy() for b in self.backstress],
                'ts': self.total_strain.copy(), 'r': self.R, 'p': self.p
            }
            trial_strain = self.total_strain + initial_guess_strain_tensor
            computed_stress = self.compute_stress(trial_strain)
            residual_at_guess = computed_stress - target_stress_tensor
            print(f"Residual at initial guess: {np.max(np.abs(residual_at_guess)):.6f} MPa")
            
            # Restore state
            self.plastic_strain = backup_state['ps']
            self.backstress = backup_state['bs']
            self.total_strain = backup_state['ts']
            self.R = backup_state['r']
            self.p = backup_state['p']
        except Exception as e:
            print(f"Error evaluating initial guess: {e}")
        
        print("=== END DIAGNOSTIC ===\n")
        
    def _monitor_convergence(self, iteration, residual_norm, delta_strain_norm, method="unknown"):
        """
        Monitor convergence during stress control iterations.
        """
        if hasattr(self, '_convergence_history'):
            self._convergence_history.append({
                'iter': iteration,
                'residual': residual_norm,
                'delta_strain': delta_strain_norm,
                'method': method
            })
        else:
            self._convergence_history = [{
                'iter': iteration,
                'residual': residual_norm,
                'delta_strain': delta_strain_norm,
                'method': method
            }]
            
        # Print convergence info for debugging
        if iteration % 5 == 0:
            print(f"  Iter {iteration}: ||R||={residual_norm:.2e}, ||Δε||={delta_strain_norm:.2e} [{method}]")
            
    def print_convergence_summary(self):
        """
        Print summary of convergence history.
        """
        if hasattr(self, '_convergence_history') and self._convergence_history:
            print("\n=== CONVERGENCE SUMMARY ===")
            history = self._convergence_history
            print(f"Total iterations: {len(history)}")
            print(f"Final residual: {history[-1]['residual']:.2e}")
            print(f"Final method: {history[-1]['method']}")
            print(f"Convergence rate: {history[-1]['residual']/history[0]['residual']:.2e}")
            print("===========================\n")
            # Clear history
            self._convergence_history = [] 

class UnifiedMaterialSolverUniAxial:
    """
    Advanced unified material solver that can switch between explicit and implicit methods.
    Stand-alone solver that combines both explicit and implicit functionality.
    """
    def __init__(self, E, yield_stress, material_model, method='explicit', precision='standard'):
        """
        Initialize the advanced solver.
        
        Parameters:
        - E: Young's modulus
        - yield_stress: Initial yield stress
        - material_model: Material model object
        - method: 'explicit', 'implicit' (semi-implicit), or 'fully_implicit'
        - precision: 'standard' or 'high' or 'scientific'
        """
        self.E = E
        self.sigma_y = yield_stress
        self.material_model = material_model
        self.n_components = getattr(material_model, 'n_components', 1)
        self.method = method
        self.precision = precision
        
        # Flag to control when plastic strain history should be stored
        self._store_plastic_strain_history = True
        
        # Set tolerance based on precision level
        self._setup_method_parameters()
        self.reset_state()
    
    def reset_state(self):
        """Reset all state variables to initial conditions."""
        self.plastic_strain = 0.0
        self.backstress = np.zeros(self.n_components)
        self.current_stress = 0.0
        self.total_strain = 0.0
        self.R = 0.0
        self.p = 0.0
        self.dp_history = []
        self.plastic_strain_history = []
        
        # Reset plastic strain smoothing state
        # self._previous_plastic_strain = None # Removed smoothing state
    
    def _setup_method_parameters(self):
        """Setup parameters based on current method and precision."""
        if self.method in ['implicit', 'fully_implicit']:
            # Use more reasonable precision for implicit method to avoid oscillations
            if self.precision == 'scientific':
                self._yield_tolerance = 1e-10  # Reduced from 1e-12
            elif self.precision == 'high':
                self._yield_tolerance = 1e-9   # Reduced from 1e-10
            else:  # standard
                self._yield_tolerance = 1e-8
        else:  # explicit
            # Use standard precision for explicit method
            if self.precision == 'scientific':
                self._yield_tolerance = 1e-10
            elif self.precision == 'high':
                self._yield_tolerance = 1e-9
            else:  # standard
                self._yield_tolerance = 1e-8
            
        # **NEW: Unified stress control tolerances**
        self._stress_control_tolerance = self._yield_tolerance
        self._stress_control_max_iter = 50
        self._stress_control_methods = ['hybr', 'lm', 'broyden1']
        self._stress_control_tolerance_levels = [
            self._stress_control_tolerance,
            self._stress_control_tolerance * 10,
            self._stress_control_tolerance * 100
        ]
    
    def set_method(self, method='explicit', precision=None):
        """
        Switch between explicit, implicit (semi-implicit), and fully_implicit methods.
        
        Parameters:
        - method: 'explicit', 'implicit', or 'fully_implicit'
        - precision: 'standard', 'high', or 'scientific' (optional)
        """
        old_method = self.method
        self.method = method
        
        if precision is not None:
            self.precision = precision
        
        self._setup_method_parameters()
        
        # Print method switching information (keep this as solver indicator)
        print(f"Switched from {old_method} to {method} method with {self.precision} precision")
    
    def compute_stress(self, total_strain):
        """
        Compute stress using the selected method.
        """
        if self.method == 'fully_implicit':
             return self._compute_stress_fully_implicit(total_strain)
        elif self.method == 'implicit':
            return self._compute_stress_implicit(total_strain)
        else:  # explicit
            return self._compute_stress_explicit(total_strain)
            
    # ... (Explicit methods unchanged) ...

    # ... (Semi-Implicit methods unchanged) ...
    
    def _compute_stress_fully_implicit(self, total_strain):
        """
        Fully Implicit stress computation.
        Identical structure to _compute_stress_implicit, but calls _solve_plastic_multiplier_fully_implicit.
        """
        # Calculate strain increment
        d_strain = total_strain - self.total_strain
        
        # If strain increment is negligible, return current stress
        if abs(d_strain) < 1e-12:
            self.dp_history.append(0.0)
            if self._store_plastic_strain_history:
                self.plastic_strain_history.append(self.plastic_strain)
            return self.current_stress
        
        # Sub-stepping specific for fully implicit (can be larger than semi-implicit)
        strain_increment_size = abs(d_strain)
        if strain_increment_size > 0.001:  # 0.1% strain (reduced from 0.02% to limit substeps)
            n_substeps = max(int(strain_increment_size / 0.001), 4)
        else:
            n_substeps = 1
        
        # Limit sub-steps to 10 maximum for efficiency
        n_substeps = min(n_substeps, 10)
        
        d_strain_sub = d_strain / n_substeps
        dp_total_step = 0.0
        
        for i in range(n_substeps):
            sub_target_strain = self.total_strain + d_strain_sub
            
            # Elastic trial step
            trial_stress = self.material_model.compute_trial_stress(sub_target_strain, self.plastic_strain, self.E)
            total_backstress = np.sum(self.backstress)
            R = getattr(self, 'R', 0.0)
            
            current_yield_stress = self.sigma_y + R
            f, shifted_stress = self.material_model.yield_function(trial_stress, total_backstress, current_yield_stress)

            effective_tolerance = max(self._yield_tolerance, 1e-10)

            if f <= effective_tolerance:
                # Elastic step
                self.current_stress = trial_stress
                self.total_strain = sub_target_strain
            else:
                # Plastic step - use FULLY IMPLICIT solver
                sign = np.sign(shifted_stress)
                
                # CALL THE NEW FULLY IMPLICIT SOLVER
                dp_solution = self._solve_plastic_multiplier_fully_implicit(trial_stress, total_backstress, sign, R)
                
                # Accumulate plastic multiplier
                dp_total_step += dp_solution
                
                # Apply solution using implicit state update (Consistent with the solver solution)
                self._update_state_variables_implicit(dp_solution, sign)
                self.current_stress = trial_stress - self.E * dp_solution * sign
                self.total_strain = sub_target_strain
        
        self.dp_history.append(dp_total_step)
        if self._store_plastic_strain_history:
            self.plastic_strain_history.append(self.plastic_strain)
        return self.current_stress

    def _solve_plastic_multiplier_fully_implicit(self, trial_stress, total_backstress_old, sign, R_old):
        """
        Solves for plastic multiplier dp such that the final state (alpha_new, R_new, sigma_new) 
        satisfies the yield function f = 0.
        CRITICAL: Uses update_backstress_implicit inside the residual.
        OPTIMIZED: Reduced iterations and improved convergence strategy.
        """
        # Cache frequently accessed values
        p_old = getattr(self, 'p', 0.0)
        backstress_current = self.backstress
        E = self.E
        sigma_y = self.sigma_y
        material_model = self.material_model
        
        def residual(dp):
            if dp < 0:
                return 1e8 * abs(dp)
            
            dp_reg = max(dp, 1e-16)

            # 1. Update Backstress IMPLICITLY (direct formula, no iteration for Chaboche)
            new_backstress = material_model.update_backstress_implicit(backstress_current, dp_reg, sign)

            # 2. Update Isotropic Hardening IMPLICITLY
            new_R, _ = material_model.isotropic_model.update_implicit(R_old, p_old, dp_reg)

            # 3. Check Yield Function at NEW state
            stress_new = trial_stress - E * dp_reg * sign
            total_backstress_new = np.sum(new_backstress)
            
            f_new, _ = material_model.yield_function(stress_new, total_backstress_new, sigma_y + new_R)
            
            return f_new
        
        # --- Optimized Solver Strategy ---
        f_initial, _ = material_model.yield_function(trial_stress, total_backstress_old, sigma_y + R_old)
        
        # Better initial guess using effective modulus
        state_vars_dummy = {'sign': sign, 'R': R_old, 'p': p_old}
        plastic_modulus = material_model.compute_plastic_modulus(backstress_current, state_vars_dummy)
        effective_modulus = E + max(plastic_modulus, 1000)
        initial_guess = max(f_initial / effective_modulus, 1e-12)

        # Strategy 1: Try Newton with better tolerance (uses scipy.optimize.newton imported at top)
        try:
            dp_solution = newton(residual, x0=initial_guess, tol=1e-8, maxiter=30)
            if dp_solution >= 0 and abs(residual(dp_solution)) < 1e-6:
                return dp_solution
        except:
            pass
        
        # Strategy 2: Try with scaled initial guesses
        for scale in [0.5, 2.0, 0.1, 5.0]:
            try:
                dp_solution = newton(residual, x0=initial_guess * scale, tol=1e-8, maxiter=20)
                if dp_solution >= 0 and abs(residual(dp_solution)) < 1e-6:
                    return dp_solution
            except:
                continue
            
        # Strategy 3: Bracketing with brentq (more efficient)
        from scipy.optimize import brentq
        try:
            # Find bracket efficiently
            val_0 = residual(0.0)
            if val_0 <= 0:
                return 0.0
            
            # Quick geometric search for upper bound
            upper = initial_guess
            for _ in range(20):  # Reduced from 50
                val = residual(upper)
                if val <= 0:
                    break
                upper *= 2.0
            else:
                # If still no bracket, try a larger jump
                upper = initial_guess * 100
                if residual(upper) > 0:
                    raise RuntimeError("Cannot find bracket")

            # Brentq is very efficient once we have a bracket
            dp_solution = brentq(residual, 0.0, upper, xtol=1e-10, maxiter=30)
            if dp_solution >= 0:
                return dp_solution
                
        except Exception as e:
            pass

        raise RuntimeError(f"Fully Implicit solver failed at f_initial={f_initial}")
    
    def _compute_stress_explicit(self, total_strain):
        """
        Explicit stress computation with sub-stepping for robustness.
        """
        # Check if we're crossing the yield point and need sub-stepping
        elastic_trial_stress = self.E * (total_strain - self.plastic_strain)
        total_backstress = np.sum(self.backstress)
        R = getattr(self, 'R', 0.0)
        current_yield_stress = self.sigma_y + R
        
        # Check if this step crosses yield point
        f_trial = abs(elastic_trial_stress - total_backstress) - current_yield_stress
        
        # If we're crossing from elastic to plastic, add an intermediate step at exact yield
        if f_trial > 1e-8 and hasattr(self, '_last_stress') and abs(self._last_stress - total_backstress) <= current_yield_stress:
            # Calculate strain that gives exact yield stress
            if elastic_trial_stress > total_backstress:
                exact_yield_strain = self.plastic_strain + (current_yield_stress + total_backstress) / self.E
            else:
                exact_yield_strain = self.plastic_strain + (-current_yield_stress + total_backstress) / self.E
            
            # First, compute stress at exact yield point (disable plastic strain storage)
            if abs(exact_yield_strain - self.total_strain) > 1e-10:
                # Temporarily disable plastic strain history storage for intermediate calculation
                self._store_plastic_strain_history = False
                yield_stress = self._compute_stress_internal_explicit(exact_yield_strain)
                # Re-enable plastic strain history storage
                self._store_plastic_strain_history = True
                # Store this intermediate result
                self._yield_point_stress = yield_stress
        
        # Now compute the actual stress for the target strain
        result = self._compute_stress_internal_explicit(total_strain)
        self._last_stress = result
        return result
    
    def _compute_stress_internal_explicit(self, total_strain):
        """
        Internal explicit stress computation method with sub-stepping.
        """
        # Calculate strain increment
        d_strain = total_strain - self.total_strain
        
        # If strain increment is negligible, return current stress
        if abs(d_strain) < 1e-12:
            # Still need to store history for plotting consistency
            self.dp_history.append(0.0)  # No plastic activity
            if self._store_plastic_strain_history:
                self.plastic_strain_history.append(self.plastic_strain)
            return self.current_stress
        
        # Determine number of sub-steps based on strain increment size
        strain_increment_size = abs(d_strain)
        if strain_increment_size > 0.001:  # 0.1% strain
            n_substeps = max(int(strain_increment_size / 0.001), 2)
        else:
            n_substeps = 1
        
        # Limit maximum sub-steps to prevent excessive computation
        n_substeps = min(n_substeps, 20)
        n_substeps = 20
        # Sub-stepping
        d_strain_sub = d_strain / n_substeps
        
        # Accumulate plastic multiplier across all sub-steps
        dp_total_step = 0.0
        
        for i in range(n_substeps):
            sub_target_strain = self.total_strain + d_strain_sub
            
            # Elastic trial step
            trial_stress = self.material_model.compute_trial_stress(sub_target_strain, self.plastic_strain, self.E)
            total_backstress = np.sum(self.backstress)

            # Get current isotropic hardening value from solver state
            R = getattr(self, 'R', 0.0)
            p = getattr(self, 'p', 0.0)
            
            # Yield check with current yield stress (sigma_y + R)
            current_yield_stress = self.sigma_y + R
            f, shifted_stress = self.material_model.yield_function(trial_stress, total_backstress, current_yield_stress)

            if f <= 1e-8:
                # Elastic step
                self.current_stress = trial_stress
                self.total_strain = sub_target_strain
                # No plastic multiplier for elastic step
            else:
                # Plastic step
                sign = np.sign(shifted_stress)
                # Pass current state to material model
                state_vars = {'sign': sign, 'R': R, 'p': p}

                # Check if plastic modulus is valid using material model method
                h_total = self.material_model.compute_plastic_modulus(self.backstress, state_vars)
                denominator = self.E + h_total
                
                if denominator <= 1e-6:
                    # Warning only, continue with elastic step
                    # print(f"Warning: Plastic modulus near zero (denominator={denominator:.2f}), using smaller step")
                    dp_total_step = 0.0
                    break
                else:
                    d_epsilon = self.material_model.plastic_multiplier(f, self.E, self.backstress, state_vars)
                
                # Limit plastic multiplier to prevent excessive increments
                d_epsilon = min(d_epsilon, 0.001)  # Limit to 0.1% strain per sub-step
                
                # Accumulate plastic multiplier for this sub-step
                dp_total_step += d_epsilon
                
                # Update kinematic hardening (backstress) using EXPLICIT methods from material model
                self.backstress = self.material_model.update_backstress(self.backstress, d_epsilon, sign, state_vars)

                # Update isotropic hardening and get the new state using EXPLICIT methods from material model
                updated_state_vars = self.material_model.update_state_vars(state_vars, d_epsilon, sign)
                self.R = updated_state_vars.get('R', self.R)
                self.p = updated_state_vars.get('p', self.p)

                # Update total plastic strain and final stress
                self.plastic_strain += d_epsilon * sign
                self.current_stress = trial_stress - self.E * d_epsilon * sign
                self.total_strain = sub_target_strain
        
        # Append total plastic multiplier once per strain step (matching multi-axial behavior)
        self.dp_history.append(dp_total_step)
        
        # Store plastic strain history for plotting (only if flag is True)
        if self._store_plastic_strain_history:
            self.plastic_strain_history.append(self.plastic_strain)
        return self.current_stress
    
    # Removed _apply_plastic_strain_smoothing_uniaxial method
    
    def _compute_stress_implicit(self, total_strain):
        """
        Simplified implicit stress computation to eliminate oscillations.
        Uses the same basic logic as explicit but with implicit state updates.
        """
        # Calculate strain increment
        d_strain = total_strain - self.total_strain
        
        # If strain increment is negligible, return current stress
        if abs(d_strain) < 1e-12:
            # Still need to store history for plotting consistency
            self.dp_history.append(0.0)  # No plastic activity
            if self._store_plastic_strain_history:
                self.plastic_strain_history.append(self.plastic_strain)
            return self.current_stress
        
        # **IMPROVED: More aggressive sub-stepping for implicit method**
        strain_increment_size = abs(d_strain)
        if strain_increment_size > 0.0005:  # 0.05% strain (more aggressive)
            n_substeps = max(int(strain_increment_size / 0.0005), 4)  # More substeps
        else:
            n_substeps = 1
        
        # **IMPROVED: Allow more sub-steps for better stability**
        n_substeps = min(n_substeps, 5000)  # Increased limit
        
        # Sub-stepping
        d_strain_sub = d_strain / n_substeps
        
        # Accumulate plastic multiplier across all sub-steps
        dp_total_step = 0.0
        
        for i in range(n_substeps):
            sub_target_strain = self.total_strain + d_strain_sub
            
            # Elastic trial step
            trial_stress = self.material_model.compute_trial_stress(sub_target_strain, self.plastic_strain, self.E)
            total_backstress = np.sum(self.backstress)

            # Get current isotropic hardening value from solver state
            R = getattr(self, 'R', 0.0)
            p = getattr(self, 'p', 0.0)
            
            # Yield check with current yield stress (sigma_y + R)
            current_yield_stress = self.sigma_y + R
            f, shifted_stress = self.material_model.yield_function(trial_stress, total_backstress, current_yield_stress)

            # Use a more reasonable tolerance to avoid oscillations
            effective_tolerance = max(self._yield_tolerance, 1e-10)  # Don't go below 1e-10

            if f <= effective_tolerance:
                # Elastic step
                self.current_stress = trial_stress
                self.total_strain = sub_target_strain
                # No plastic multiplier for elastic step
            else:
                # Plastic step - use implicit solver for plastic multiplier
                sign = np.sign(shifted_stress)
                dp_solution = self._solve_plastic_multiplier_robust(trial_stress, total_backstress, sign, R)
                
                # Accumulate plastic multiplier for this sub-step
                dp_total_step += dp_solution
                
                # Apply solution using implicit state update
                self._update_state_variables_implicit(dp_solution, sign)
                self.current_stress = trial_stress - self.E * dp_solution * sign
                self.total_strain = sub_target_strain
        
        # Append total plastic multiplier once per strain step (matching multi-axial behavior)
        self.dp_history.append(dp_total_step)
        
        # Store plastic strain history for plotting (only if flag is True)
        if self._store_plastic_strain_history:
            self.plastic_strain_history.append(self.plastic_strain)
        return self.current_stress
    
    def _solve_plastic_multiplier_robust(self, trial_stress, total_backstress, sign, R):
        """
        Simplified and more stable solver for plastic multiplier.
        """
        def residual(dp):
            if dp < 0:
                return 1e8 * abs(dp)  # Penalty for negative dp

            dp_reg = max(dp, 1e-16)

            # --- UNIFIED STATE VARIABLE UPDATE (FIXED) ---
            # Create a temporary state dictionary to pass to the material model's update functions.
            # This ensures that the residual calculation uses the exact same logic as the final state update.
            temp_state_vars = {
                'sign': sign,
                'R': R,
                'p': getattr(self, 'p', 0.0)
            }
            
            # Use the material model's own methods to get the trial state variables
            new_backstress = self.material_model.update_backstress(self.backstress, dp_reg, sign, temp_state_vars)
            new_R, _ = self.material_model.isotropic_model.update_implicit(R, temp_state_vars['p'], dp_reg)

            # --- YIELD FUNCTION ---
            stress_new = trial_stress - self.E * dp_reg * sign
            total_backstress_new = np.sum(new_backstress)
            f_new, _ = self.material_model.yield_function(stress_new, total_backstress_new, self.sigma_y + new_R)
            
            return f_new
        
        # Enhanced initial guess based on elastic predictor
        f_initial, shifted_stress_initial = self.material_model.yield_function(trial_stress, total_backstress, self.sigma_y + R)
        sign = np.sign(shifted_stress_initial)
        
        # Use a simpler, more conservative initial guess
        state_vars_for_modulus = {'sign': sign, 'R': R, 'p': getattr(self, 'p', 0.0)}
        
        # Use a simpler, more conservative initial guess
        plastic_modulus = self.material_model.compute_plastic_modulus(self.backstress, state_vars_for_modulus)
        effective_modulus = self.E + max(plastic_modulus, 1000)  # Add safety margin
        initial_guess = max(f_initial / effective_modulus, 1e-12)
        
        # Strategy 1: Newton's method (uses newton imported at top of file)
        try:
            dp_solution = newton(residual, x0=initial_guess, tol=1e-8, maxiter=25)
            if dp_solution >= 0 and abs(residual(dp_solution)) < 1e-6:
                return dp_solution
        except:
            pass
        
        # Strategy 2: Try with scaled initial guesses
        for scale in [0.5, 2.0, 0.1]:
            try:
                dp_solution = newton(residual, x0=initial_guess * scale, tol=1e-8, maxiter=20)
                if dp_solution >= 0 and abs(residual(dp_solution)) < 1e-6:
                    return dp_solution
            except:
                continue

        # Strategy 3: Bracketing solver (brentq)
        from scipy.optimize import brentq
        try:
            lower_bound = 0.0
            if residual(lower_bound) > 0:
                upper_bound = initial_guess * 2
                for _ in range(15):
                    if residual(upper_bound) <= 0:
                        break
                    upper_bound *= 2
                else:
                    raise RuntimeError("Failed to find bracket")

                dp_solution = brentq(residual, lower_bound, upper_bound, xtol=1e-10, maxiter=50)
                if dp_solution >= 0:
                    return dp_solution
        except:
            pass

        raise RuntimeError(f"All implicit plastic multiplier solvers failed at trial_stress={trial_stress:.2f}, f_initial={f_initial:.4f}")

    def _update_state_variables_implicit(self, dp, sign):
        """Update all state variables after plastic step using implicit method."""
        # Use material model's own implicit update for backstress
        if hasattr(self.material_model, 'update_backstress_implicit'):
            self.backstress = self.material_model.update_backstress_implicit(self.backstress, dp, sign)
        else:
            # fallback: use explicit update if no implicit version is provided
            self.backstress = self.material_model.update_backstress(self.backstress, dp, sign, {'sign': sign, 'R': getattr(self, 'R', 0.0), 'p': getattr(self, 'p', 0.0)})
        # Update isotropic hardening using implicit method through isotropic model (legacy version for consistency)
        R_current = getattr(self, 'R', 0.0)
        p_current = getattr(self, 'p', 0.0)
        R_new, p_new = self.material_model.isotropic_model.update_implicit(R_current, p_current, dp)
        self.R = R_new
        self.p = p_new
        # Update total plastic strain
        self.plastic_strain += dp * sign
    
    def compute_strain(self, target_stress, tol=None, max_iter=None):
        """
        Stress-controlled strain update. Works for both explicit and implicit methods.
        Enhanced with unified convergence handling for better consistency between solvers.
        """
        # **NEW: Use unified tolerance settings**
        if tol is None:
            tol = self._stress_control_tolerance
        if max_iter is None:
            max_iter = self._stress_control_max_iter
        
        # **NEW: Unified residual function with enhanced error handling**
        def residual(delta_strain):
            # Backup all state variables before trial
            backup_plastic = self.plastic_strain
            backup_backstress = self.backstress.copy()
            backup_total = self.total_strain
            backup_R = self.R
            backup_p = self.p
            backup_dp_history = self.dp_history.copy()
            backup_plastic_strain_history = self.plastic_strain_history.copy()
            backup_current_stress = self.current_stress  # BUG FIX: Also backup current_stress

            # Disable plastic strain history storage during iterations
            self._store_plastic_strain_history = False

            try:
                trial_strain = self.total_strain + delta_strain
                computed_stress = self.compute_stress(trial_strain)
                residual_value = computed_stress - target_stress
            except Exception as e:
                # Return large penalty for failed computation
                residual_value = 1e8 * abs(delta_strain)

            # Restore all state variables after trial
            self.plastic_strain = backup_plastic
            self.backstress = backup_backstress.copy()
            self.total_strain = backup_total
            self.R = backup_R
            self.p = backup_p
            self.dp_history = backup_dp_history
            self.plastic_strain_history = backup_plastic_strain_history
            self.current_stress = backup_current_stress  # BUG FIX: Restore current_stress
            
            return residual_value
        
        # **IMPROVED: Enhanced initial guess strategy (unified with multiaxial)**
        delta_stress = target_stress - self.current_stress
        delta_strain_elastic = delta_stress / self.E
        
        # **NEW: Account for current backstress in initial guess**
        total_backstress = np.sum(self.backstress)
        effective_stress_change = (target_stress - total_backstress) - (self.current_stress - total_backstress)
        
        # **UNIFIED: Same adaptive scaling logic as multiaxial solver**
        stress_magnitude = abs(target_stress)
        if stress_magnitude > 100:  # High stress level
            scaling_factor = 0.3
        elif stress_magnitude > 10:  # Medium stress level
            scaling_factor = 0.5
        else:  # Low stress level
            scaling_factor = 0.7
        
        # **UNIFIED: Initial guess strategy matching multiaxial solver**
        initial_guess_strain = delta_strain_elastic * scaling_factor
        
        # **UNIFIED: Use root method with multiple attempts (same as multiaxial solver)**
        solution_converged = False
        final_delta_strain = None
        
        # OPTIMIZED: Simplified iteration strategy for better performance
        # Uses fewer attempts but with better initial guesses
        
        # Strategy 1: Try Newton with the best initial guess first
        try:
            # Use scipy.optimize.newton for simple 1D problem (faster than root)
            dp_solution = newton(residual, x0=initial_guess_strain, tol=tol, maxiter=30)
            if abs(residual(dp_solution)) < tol * 10 and abs(dp_solution) < 0.1:
                final_delta_strain = dp_solution
                solution_converged = True
        except:
            pass
        
        # Strategy 2: Try with elastic estimate
        if not solution_converged:
            try:
                dp_solution = newton(residual, x0=delta_strain_elastic, tol=tol, maxiter=30)
                if abs(residual(dp_solution)) < tol * 10 and abs(dp_solution) < 0.1:
                    final_delta_strain = dp_solution
                    solution_converged = True
            except:
                pass
        
        # Strategy 3: Try with scaled guesses
        if not solution_converged:
            for scale in [0.5, 2.0, 0.1]:
                try:
                    dp_solution = newton(residual, x0=initial_guess_strain * scale, tol=tol * 10, maxiter=25)
                    if abs(residual(dp_solution)) < tol * 100 and abs(dp_solution) < 0.1:
                        final_delta_strain = dp_solution
                        solution_converged = True
                        break
                except:
                    continue
        
        # Strategy 4: Bracketing with brentq (robust fallback)
        if not solution_converged:
            from scipy.optimize import brentq
            try:
                # Find bracket
                val_0 = residual(0.0)
                sign_strain = 1.0 if delta_strain_elastic > 0 else -1.0
                
                if delta_strain_elastic > 0:
                    # Positive strain increment
                    upper = abs(delta_strain_elastic) * 3
                    for _ in range(15):
                        if residual(upper) * val_0 < 0:
                            break
                        upper *= 2
                    else:
                        upper = abs(delta_strain_elastic) * 10
                    
                    if residual(upper) * val_0 < 0:
                        final_delta_strain = brentq(residual, 0.0, upper, xtol=tol, maxiter=50)
                        solution_converged = True
                else:
                    # Negative strain increment
                    lower = delta_strain_elastic * 3
                    for _ in range(15):
                        if residual(lower) * val_0 < 0:
                            break
                        lower *= 2
                    else:
                        lower = delta_strain_elastic * 10
                    
                    if residual(lower) * val_0 < 0:
                        final_delta_strain = brentq(residual, lower, 0.0, xtol=tol, maxiter=50)
                        solution_converged = True
            except:
                pass
        
        # Final fallback: use elastic estimate with warning
        if not solution_converged:
            stress_diff = abs(target_stress - self.current_stress)
            if stress_diff < 1.0:
                final_delta_strain = initial_guess_strain * 0.1
                solution_converged = True
            else:
                # Use best available estimate
                final_delta_strain = delta_strain_elastic
                solution_converged = True
        
        # **CRITICAL FIX: Final solution application with stress consistency**
        final_strain = self.total_strain + final_delta_strain
        
        # Re-enable plastic strain history storage for final computation
        self._store_plastic_strain_history = True
        
        # **IMPORTANT: Let compute_stress determine the actual stress**
        actual_stress = self.compute_stress(final_strain)
        self.current_stress = actual_stress  # Use computed stress, not forced target
        
        # **NEW: Check if we're close enough to target**
        stress_error = abs(actual_stress - target_stress)
        if stress_error > 1.0:  # If error > 1 MPa
            print(f"Warning: Stress error {stress_error:.2f} MPa at target {target_stress:.2f} MPa")

        return self.total_strain
    
    def run_strain_controlled(self, strain_history):
        """
        Strain-controlled simulation with proper state tracking.
        """
        stresses = []
        strains = []
        for strain in strain_history:
            stress = self.compute_stress(strain)
            stresses.append(stress)
            strains.append(strain)
        return np.array(stresses), np.array(strains)

    def run_stress_controlled(self, stress_history):
        """
        Stress-controlled simulation with enhanced stability.
        """
        strains = []
        stresses = []
        for stress in stress_history:
            try:
                strain = self.compute_strain(stress)
                strains.append(strain)
                # **CRITICAL FIX: Use actual computed stress, not target stress**
                stresses.append(self.current_stress)  # Use actual stress from solver
            except RuntimeError as e:
                # print(f"Simulation stopped: {str(e)}")
                break
        return np.array(strains), np.array(stresses)

    def generate_cyclic_path(self, amp_pos, amp_neg, n_cycles, n_points):
        """
        Generate a ratcheting loading path and return the path, cycle_ids, and sequence_ids.
        Each cycle is defined as one unloading-loading pair (except the first, which is loading-unloading-loading).
        Returns:
            path: ndarray of strain or stress values
            cycle_ids: ndarray of cycle index for each point
            sequence_ids: ndarray of segment index for each point
        """
        path = []
        cycle_ids = []
        sequence_ids = []
        seq = 0
        # First cycle: loading (0->+), unloading (+->-), loading (- -> +)
        seg = np.linspace(0, amp_pos, n_points)
        path.extend(seg)
        cycle_ids.extend([0]*n_points)
        sequence_ids.extend([seq]*n_points)
        seq += 1
        seg = np.linspace(amp_pos, amp_neg, n_points)
        path.extend(seg)
        cycle_ids.extend([0]*n_points)
        sequence_ids.extend([seq]*n_points)
        seq += 1
        seg = np.linspace(amp_neg, amp_pos, n_points)
        path.extend(seg)
        cycle_ids.extend([0]*n_points)
        sequence_ids.extend([seq]*n_points)
        seq += 1
        # Subsequent cycles: unloading (+->-), loading (- -> +)
        for c in range(1, n_cycles):
            seg = np.linspace(amp_pos, amp_neg, n_points)
            path.extend(seg)
            cycle_ids.extend([c]*n_points)
            sequence_ids.extend([seq]*n_points)
            seq += 1
            seg = np.linspace(amp_neg, amp_pos, n_points)
            path.extend(seg)
            cycle_ids.extend([c]*n_points)
            sequence_ids.extend([seq]*n_points)
            seq += 1
        return np.array(path), np.array(cycle_ids), np.array(sequence_ids)
    
    def get_method_info(self):
        """Return information about current method and settings."""
        return {
            'method': self.method,
            'precision': self.precision,
            'yield_tolerance': self._yield_tolerance,
            'available_methods': ['explicit', 'implicit', 'fully_implicit'],
            'available_precisions': ['standard', 'high', 'scientific']
        } 

    def diagnose_stress_control(self, target_stress):
        """
        Diagnostic tool for stress control convergence issues.
        Analyzes the residual function behavior around the current state.
        """
        print("=== UNIAXIAL STRESS CONTROL DIAGNOSTIC ===")
        print(f"Target stress: {target_stress:.3f} MPa")
        print(f"Current stress: {self.current_stress:.3f} MPa")
        print(f"Stress difference: {abs(target_stress - self.current_stress):.3f} MPa")
        
        # Test initial guess quality
        delta_stress = target_stress - self.current_stress
        initial_guess_strain = delta_stress / self.E
        total_backstress = np.sum(self.backstress)
        effective_stress_change = (target_stress - total_backstress) - (self.current_stress - total_backstress)
        
        print(f"Elastic initial guess: {initial_guess_strain:.6f}")
        print(f"Total backstress: {total_backstress:.3f} MPa")
        print(f"Effective stress change: {effective_stress_change:.3f} MPa")
        
        # Test residual at initial guess
        try:
            backup_state = {
                'ps': self.plastic_strain, 'bs': self.backstress.copy(),
                'ts': self.total_strain, 'r': self.R, 'p': self.p
            }
            trial_strain = self.total_strain + initial_guess_strain
            computed_stress = self.compute_stress(trial_strain)
            residual_at_guess = computed_stress - target_stress
            print(f"Residual at initial guess: {residual_at_guess:.6f} MPa")
            
            # Restore state
            self.plastic_strain = backup_state['ps']
            self.backstress = backup_state['bs']
            self.total_strain = backup_state['ts']
            self.R = backup_state['r']
            self.p = backup_state['p']
        except Exception as e:
            print(f"Error evaluating initial guess: {e}")
        
        print("=== END DIAGNOSTIC ===\n")
        
    def _monitor_convergence(self, iteration, residual_norm, delta_strain_norm, method="unknown"):
        """
        Monitor convergence during stress control iterations.
        """
        if hasattr(self, '_convergence_history'):
            self._convergence_history.append({
                'iter': iteration,
                'residual': residual_norm,
                'delta_strain': delta_strain_norm,
                'method': method
            })
        else:
            self._convergence_history = [{
                'iter': iteration,
                'residual': residual_norm,
                'delta_strain': delta_strain_norm,
                'method': method
            }]
            
        # Print convergence info for debugging
        if iteration % 5 == 0:
            print(f"  Iter {iteration}: ||R||={residual_norm:.2e}, ||Δε||={delta_strain_norm:.2e} [{method}]")
            
    def print_convergence_summary(self):
        """
        Print summary of convergence history.
        """
        if hasattr(self, '_convergence_history') and self._convergence_history:
            print("\n=== CONVERGENCE SUMMARY ===")
            history = self._convergence_history
            print(f"Total iterations: {len(history)}")
            print(f"Final residual: {history[-1]['residual']:.2e}")
            print(f"Final method: {history[-1]['method']}")
            print(f"Convergence rate: {history[-1]['residual']/history[0]['residual']:.2e}")
            print("===========================\n")
            # Clear history
            self._convergence_history = [] 