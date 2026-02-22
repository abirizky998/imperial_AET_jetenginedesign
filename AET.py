import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import simpson
import os

from itertools import product

### Emergency Hover and Propeller Sizing ###
# max thrust should allow one engine to hover the aircraft such that V_y = 0
def max_thrust(mass, acc, g): # total thrust 
    return mass * g

def propeller_sizing(T_Max, density, V_e, delta_pressure): # assuming in test bed or during single engine hover
    T_prop_gross = T_Max * 0.95 # assuming 95% of thrust comes from the prop
    Area = T_prop_gross / (delta_pressure + 0.5 * density * V_e ** 2)
    return np.sqrt(4 * Area / np.pi)

def prop_max_power(T_Max, density, prop_eff, dp):
    return np.sqrt((2 * (T_Max ** 3)) / (np.pi * density * prop_eff**2 * dp**2))
    
max_weight = 19000 #kg
gravity = 9.81 #m/s^2
downwards_acc = 0.5
acc = np.linspace(0.1, 5, 500)
height = 1500 #m, roughly 5k ft
air_density = 1.05 #  np.linspace(1.05, 1.225, 100) # range between sea level to 1500 m

# takeoff_kinematics(acc, height)
thrust_maximum = max_thrust(max_weight, downwards_acc, gravity)
prop_diameter = propeller_sizing(thrust_maximum, air_density, 20, 1500)
P = prop_max_power(thrust_maximum, air_density, 0.80, prop_diameter)

print(f"T_max = {thrust_maximum}, D_p = {prop_diameter}, P = {P}")

def ISA_atmosphere(altitude):
    g = 9.81 # m/s^2
    R = 8.314 # J/(mol K)
    M = 0.02896 # kg/mol
    P0 = 101325.0 # Pa
    T0 = 288.15 # K
    L = 0.0065 # K/m, temperature gradient w.r.t alt

    altitude = np.asarray(altitude)
    T = T0 - (L * altitude)
    P = P0 * (1 - (L * altitude) / T0) ** (g * M / (R * L))
    rho = (P * M) / (R * T)
    
    return P, T, rho

def isentropic_calc(p_in, T_in, pressure_ratio, gamma, turbo):
    if turbo == "comp":
        return p_in * pressure_ratio, T_in * pressure_ratio ** ((gamma-1) / gamma)
    elif turbo == "turb":
        return p_in / pressure_ratio, T_in / (pressure_ratio ** ((gamma-1) / gamma))

def enthalpy_calc(Cp, T_high, T_low):
    return abs(Cp * (T_high - T_low))

def compressor(p1, T1, pressure_ratio, gamma, Cp):
    p2, T2 = isentropic_calc(p1, T1, pressure_ratio, gamma, "comp")
    work = 0.85 * enthalpy_calc(Cp, T2, T1)
    return work, p2, T2

def combustor(target_T3, T2, P2, pressure_loss, Cp):
    heat = enthalpy_calc(Cp, target_T3, T2)
    return heat, P2 * pressure_loss

def turbine(p3, target_T3, pressure_ratio, gamma, Cp):
    p4, T4 = isentropic_calc(p3, target_T3, pressure_ratio, gamma, "turb")
    work = 0.89 * enthalpy_calc(Cp, target_T3, T4)
    return work, p4, T4

def brayton_cycle_analysis(alt_list, tit_list, pr_list):
    gamma_c, Cp_c = 1.4, 1.005
    gamma_h, Cp_h = 1.33, 1.15
    combustor_eff = 0.95
    results = []

    for altitude, T3, pressure_ratio in product(alt_list, tit_list, pr_list):
        P1, T1, rho1 = ISA_atmosphere(altitude)
        work_comp, P2, T2 = compressor(P1, T1, pressure_ratio, gamma_c, Cp_c)
        heat_in, P3 = combustor(T3, T2, P2, combustor_eff, Cp_h)
        expansion_ratio = P3 / P1
        work_turb, P4, T4 = turbine(P3, T3, expansion_ratio, gamma_h, Cp_h)

        work_net = work_turb - work_comp
        sfc = (heat_in / work_net) * 3.6e6 / 43.1e6

        results.append({
            "Cruise_Altitude": altitude,
            "TIT (K)": T3,
            "PR": pressure_ratio,
            "Availabe Specific Propeller Power (kJ / kg)": work_net,
            "Heat In (kJ / kg)": heat_in,
            "SFC (kg / kWh)": sfc,
            "Inlet Air Density (kg/m^3)": rho1
        })
        
    return pd.DataFrame(results)

def performance_plot(df, design_altitude):
    subset = df[df['Cruise_Altitude'] == design_altitude].copy()
    plt.style.use('bmh')
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"Engine Cycle Performance @ {design_altitude}m Altitude", fontsize=16)

    # Pressure Ratio vs. SFC
    ax1 = fig.add_subplot(2, 2, 1)
    scatter1 = ax1.scatter(
        subset['PR'], 
        subset['SFC (kg / kWh)'], 
        c=subset['TIT (K)'], 
        cmap='inferno', 
        s=50, 
        edgecolor='k'
    )
    ax1.set_xlabel("Pressure Ratio")
    ax1.set_ylabel("SFC (kg/kWh)")
    ax1.set_title("Pressure Ratio and Turbine Inlet Temperature Effects on SFC")
    fig.colorbar(scatter1, ax=ax1, label='TIT (K)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Design Space Heatmap (TIT vs PR)
    ax2 = fig.add_subplot(2, 2, 2)
    pivot_table = subset.pivot_table(values='SFC (kg / kWh)', index='TIT (K)', columns='PR')
    contour = ax2.contourf(
        pivot_table.columns, 
        pivot_table.index, 
        pivot_table.values, 
        levels=20, 
        cmap='RdYlGn_r'
    )
    ax2.set_xlabel("Pressure Ratio")
    ax2.set_ylabel("Turbine Inlet Temp (K)")
    ax2.set_title("Iso-SFC Contours (Green is Better)")
    fig.colorbar(contour, ax=ax2, label='SFC (kg/kWh)')


    # (Specific Power vs. SFC)
    ax3 = fig.add_subplot(2, 1, 2)
    tits = sorted(subset['TIT (K)'].unique())
    for t in tits:
        line_data = subset[subset['TIT (K)'] == t]
        ax3.plot(line_data['Availabe Specific Propeller Power (kJ / kg)'], 
                line_data['SFC (kg / kWh)'], 
                color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax3.text(line_data['Availabe Specific Propeller Power (kJ / kg)'].iloc[-1], 
                line_data['SFC (kg / kWh)'].iloc[-1], 
                f'TIT={t}K', fontsize=9)

    prs = sorted(subset['PR'].unique())
    for p in prs:
        if p % 2 == 0: 
            line_data = subset[subset['PR'] == p]
            ax3.plot(line_data['Availabe Specific Propeller Power (kJ / kg)'], 
                    line_data['SFC (kg / kWh)'], 
                    color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
    scatter3 = ax3.scatter(
        subset['Availabe Specific Propeller Power (kJ / kg)'], 
        subset['SFC (kg / kWh)'], 
        c=subset['TIT (K)'], 
        cmap='inferno', 
        s=60, 
        zorder=3
    )

    ax3.set_xlabel("Specific Shaft Power (kW / kg/s)")
    ax3.set_ylabel("SFC (kg/kWh)")
    ax3.set_title("Specific Shaft Power vs SFC")
    ax3.invert_yaxis() 
    fig.colorbar(scatter3, ax=ax3, label='TIT (K)')

    plt.tight_layout()
    plt.show()

def prop_sizing(required_shaft_power, cycle_data, density, v_face):
    sp_power = cycle_data['Available Specific Propeller Power (kJ / kg)']
    mdot_req = required_shaft_power / sp_power
    area = mdot_req / (density * v_face)
    diameter = 2 * np.sqrt(area / np.pi)
    
    return mdot_req, diameter

def prop_design_hover(T_max, D, blades=3, alpha_deg=5.0, Cl = 1.1):
    R = D / 2
    A = np.pi * R ** 2
    a = 340 # speed of sound
    v_tip = 0.7 * a
    rho = 1.2
    print(v_tip)

    n_rps = v_tip / (np.pi * D)
    n_rpm = n_rps * 60
    
    # calculating downwash velocity to calculate blade angle addition factor, phi
    # based on momentum equation, T = 0.5 rho A v^2
    v_i = np.sqrt(T_max / (2 * rho * A))
    r_stations = np.linspace(0.1 * R, R, 20)
    v_t = 2 * np.pi * n_rps * r_stations
    phi = np.arctan(v_i / v_t)

    # alpha = AoA of root, beta = AoA of tip
    alpha_rad = np.radians(alpha_deg)
    beta_rad = phi + alpha_rad
    beta_deg = np.degrees(beta_rad)

    # W is relative velocity in each stations
    # blade calculation using Blade Element Momentum Theory
    W = np.sqrt(v_i ** 2 + v_t **2)
    psi = np.arctan(v_i / v_t)
    chord_ideal = (8 * np.pi * r_stations * v_i **2) / (blades * W**2 * Cl * np.cos(psi))

    # taking the 75% radius as a reference station, as significant portion of the thrust
    # is from the outer 25% of the blade, and finally linearising the taper
    idx_75 = np.argmin(np.abs(r_stations - 0.75*R))
    chord_75 = chord_ideal[idx_75]
    chord_tip = chord_ideal[-1]
    slope = (chord_tip - chord_75) / (r_stations[-1] - r_stations[idx_75])

    chord_real = slope * (r_stations - r_stations[-1]) + chord_tip
    chord_real = np.minimum(chord_real, 0.85)

    # taking the integral to calculate the force through the blade
    dT_ideal = blades * 0.5 * rho * W**2 * chord_ideal * Cl * np.cos(phi)
    dT_real = blades * 0.5 * rho * W**2 * chord_real * Cl * np.cos(phi)

    T_ideal = simpson(dT_ideal, x=r_stations) # taking the integrals
    T_real = simpson(dT_real, x=r_stations)
    T_loss = T_ideal - T_real
    T_loss_pct = (T_loss / T_ideal) * 100

    print("\n--- Main Propeller Design ---")
    print(f"Prop Diameter: {D:.2f} m")
    print(f"Single Engine Failure RPM: {n_rpm:.0f} RPM")
    print(f"Induced Wash (v_i): {v_i:.2f} m/s")
    print(f"Max Tip Speed:  {v_tip:.1f} m/s (Mach 0.7)")
    print(f"Root Chord, Ideal:     {chord_ideal[0]:.3f} m")
    print(f"Root Chord, Real:     {chord_real[0]:.3f} m")
    print(f"Tip Chord, Ideal:      {chord_ideal[-1]:.3f} m")
    print(f"Tip Chord, Real:      {chord_real[-1]:.3f} m")

    print("\n--- Forward Integration Results ---")
    print(f"Target Thrust:      {T_max/1000:.1f} kN")
    print(f"Ideal Blade Thrust: {T_ideal/1000:.1f} kN")
    print(f"Practical Thrust:   {T_real/1000:.1f} kN")
    print(f"Thrust Penalty:     {T_loss/1000:.1f} kN ({T_loss_pct:.2f}%)")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

    ax1.plot(r_stations, chord_ideal, 'r--', linewidth=2, label='BEMT Chord')
    ax1.plot(r_stations, chord_real, 'b-', linewidth=2, label='Truncated Chord')
    ax1.set_ylabel('Required Chord Width (m)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(r_stations, beta_deg, 'g-o', linewidth=2, label='Twist Angle')
    ax1_twin.set_ylabel('Twist Angle, Beta (°)', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')

    ax1.set_title(f'Chord and Twist Distribution')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    ax2.plot(r_stations, dT_ideal / 1000, 'r--', label='BEMT')
    ax2.plot(r_stations, dT_real / 1000, 'b-', linewidth=2, label='Truncated Chord')
    ax2.fill_between(r_stations, dT_ideal / 1000, dT_real / 1000, color='red', alpha=0.2, label='Lost Thrust vs Ideal')
    ax2.set_xlabel('Radius (m)')
    ax2.set_ylabel('Section Thrust (kN/m)')
    ax2.set_title('Thrust Distribution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

    results_dict = {"Stations Radius:" : r_stations,
                    "Station Twist Angle:" : beta_deg,
                    "Station Chord Length:" :chord_real}
    return results_dict

def prop_design_mission(D, altitude, condition, v_inf, blades=3, alpha_deg=5.0, hover_results=None):
    R = D/2
    P, T, rho = ISA_atmosphere(altitude)
    a = np.sqrt(1.4 * 287.0 * T)
    v_tip = 0.7 * a

    if v_inf >= v_tip:
        raise ValueError("Forward Speed > Max Tip Mach Number")
    
    n_rps = np.sqrt(v_tip ** 2 - v_inf ** 2) / (np.pi * D)
    n_rpm = n_rps * 60
    r_stations = np.linspace(0.1 * R, R, 20)
    v_t = 2 * np.pi * n_rps * r_stations
    phi = np.arctan(v_inf / v_t)

    alpha = np.radians(alpha_deg)
    beta_rad = phi + alpha
    beta_deg = np.degrees(beta_rad)

    print(f"\n--- {condition} Propeller Design ({altitude} m) ---")
    print(f"Target V_inf:      {v_inf:.1f} m/s (Mach {v_inf/a:.3f})")
    print(f"{condition} RPM:   {n_rpm:.0f} RPM (Reduced from Hover to keep Mach 0.7)")
    print(f"Total Tip Speed:   {v_t[-1]:.1f} m/s (Mach {v_t[-1]/a})")
    print(f"Air Density:       {rho:.3f} kg/m^3")

    if hover_results is not None:
        beta_hover = hover_results['Station Twist Angle:']
        idx_75 = np.argmin(np.abs(r_stations - 0.75 * R))
        shift_ang = (beta_hover[idx_75] - beta_deg[idx_75]).item()

        print("\n--- Variable Pitch Mechanism ---")
        print(f"75% Station Cruise Angle: {beta_deg[idx_75]:.1f} deg")
        print(f"75% Station Hover Angle:  {beta_hover[idx_75]:.1f} deg")
        print(f"Required Pitch Shift: {shift_ang:.1f} degrees ({condition} -> Hover)")
    else:
        pass
    
    results_dict = {"Twist Angle (Deg):" : beta_deg}
    
    return results_dict, n_rpm, shift_ang

def evaluate_manufactured_hover(T_max, D, hover_results, cruise_results, shift_ang, blades=3):
    r_stations = hover_results["Stations Radius:"]
    chord_real = hover_results["Station Chord Length:"]
    beta_cruise = cruise_results["Twist Angle (Deg):"]
    
    # Hover Kinematics
    rho = 1.2
    R = D / 2
    A = np.pi * R ** 2
    a = 340 
    v_tip = 0.7 * a
    n_rps = v_tip / (np.pi * D)
    
    v_i = np.sqrt(T_max / (2 * rho * A))
    v_t = 2 * np.pi * n_rps * r_stations
    phi = np.arctan(v_i / v_t)
    W = np.sqrt(v_i ** 2 + v_t **2)
    
    psi = np.arctan(v_i / v_t)
    chord_ideal = (8 * np.pi * r_stations * v_i **2) / (blades * W**2 * 1.1 * np.cos(psi))
    beta_mfg = beta_cruise + shift_ang
    alpha_rad_mfg = np.radians(beta_mfg) - phi
    alpha_deg_mfg = np.degrees(alpha_rad_mfg)

    # clipping Cl values
    Cl_mfg = 0.11 * (alpha_deg_mfg + 5.0)
    Cl_mfg = np.clip(Cl_mfg, -0.2, 1.4) 

    # taking the spanwise integral for thrust
    dT_ideal = blades * 0.5 * rho * W**2 * chord_ideal * 1.1 * np.cos(phi)
    dT_trunc = blades * 0.5 * rho * W**2 * chord_real * 1.1 * np.cos(phi)
    dT_mfg = blades * 0.5 * rho * W**2 * chord_real * Cl_mfg * np.cos(phi)
    T_ideal = simpson(dT_ideal, x=r_stations)
    T_trunc = simpson(dT_trunc, x=r_stations)
    T_mfg = simpson(dT_mfg, x=r_stations)

    loss_trunc_pct = ((T_ideal - T_trunc) / T_ideal) * 100
    loss_mfg_pct = ((T_ideal - T_mfg) / T_ideal) * 100
    additional_twist_penalty = loss_mfg_pct - loss_trunc_pct

    print("\n--- Final OEI Hover Aerodynamic Evaluation ---")
    print(f"Target Thrust:                 {T_max/1000:.1f} kN")
    print(f"1. Ideal BEMT Thrust:     {T_ideal/1000:.1f} kN")
    print(f"2. Target Twist Angle:  {T_trunc/1000:.1f} kN (-{loss_trunc_pct:.1f}% loss)")
    print(f"3. Cruise Twist Angle + Pitch:   {T_mfg/1000:.1f} kN (-{loss_mfg_pct:.1f}% total loss)")
    print(f"-> Penalty solely from twist mismatch: {additional_twist_penalty:.1f}%")

    plt.figure(figsize=(8, 4))
    plt.plot(r_stations, dT_ideal / 1000, 'r--', linewidth=2, label='BEMT')
    plt.plot(r_stations, dT_trunc / 1000, 'b-.', linewidth=2, label='Truncated Chord')
    plt.plot(r_stations, dT_mfg / 1000, 'g-o', linewidth=2, label=f'Cruise-Optimised, {shift_ang}° Pitch')
    plt.fill_between(r_stations, dT_trunc / 1000, dT_mfg / 1000, color='green', alpha=0.2, label='Pitched Cruise-Optimised Blade Recovered Thrust')
    
    plt.xlabel('Radius (m)')
    plt.ylabel('Section Thrust (kN/m)')
    plt.title('Thrust Distribution: Updated with Cruise-Optimised Blade')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


