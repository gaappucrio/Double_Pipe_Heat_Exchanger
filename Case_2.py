import streamlit as st
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from seaborn.palettes import blend_palette

st.title('TROCAL Simulator - Simulation of heat transfer in a double pipe heat exchanger')
st.write('This is a simulator of a double pipe heat exchanger operating in parallel flow. When running the simulation, you will be able to visualize the temperature profile of fluids 1 (cold) and 2 (warm) as time progresses. You will also be able to view the plot of the temperature variation of fluids 1 and 2 when the heat exchanger reaches steady state.')
st.write('Below is an illustrative figure of this heat exchanger, created by the authors.')
st.image('Case 2.png', use_column_width=True)
st.write('This type of heat exchanger is commonly used in the chemical, food, and oil & gas industries.')
st.write('This simulator employs the following energy balance equations for the cold and warm fluids, based on the principle of energy conservation:')
st.image('Equations Case 2.jpg', use_column_width=True)
st.write('ATENTION: At the end of this page, you will also find a button that runs the simulation using a predefined example (“Run standard example”). This example takes around 40 seconds to run, depending on your connection speed. If you would like to use your own values, use the “Run simulation” button, and it is recommended to choose a number of nodes between 10 and 30, depending on the specific case. In addition, the inlet temperature of the cold fluid must be lower than the inlet temperature of the hot fluid; otherwise, the model may not function correctly.')


# Criando a figura para o gráfico em regime permanente
fig_permanente = plt.figure(figsize=(8, 6))

def run_simulation(L, r1, r2, n, m1, Cp1, rho1, m2, Cp2, rho2, T1i, T2i, T0, U, dx, t_final, dt):
    Ac1 = np.pi * r1**2
    Ac2 = np.pi * (r2**2-r1**2)

    x = np.linspace(dx/2, L-dx/2, n)
    T1 = np.ones(n) * T1i
    T2 = np.ones(n) * T2i
    t = np.arange(0, t_final, dt)
    
    # Função que define a EDO para a variação da temperatura para o Fluido 1
    def dT1dt_function(T1, t):
        dT1dt = np.zeros(n)
        dT1dt[1:n] = (m1 * Cp1 * (T1[0:n-1] - T1[1:n]) + U * 2 * np.pi * r1 * dx * (T2[1:n] - T1[1:n])) / (rho1 * Cp1 * dx * Ac1)
        dT1dt[0] = (m1 * Cp1 * (T1i - T1[0]) + U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1)
        return dT1dt
    
    # Função que define a EDO para a variação da temperatura para o Fluido 2
    def dT2dt_function(T2, t):
        dT2dt = np.zeros(n)
        dT2dt[1:n] = (m2 * Cp2 * (T2[0:n-1] - T2[1:n]) - U * 2 * np.pi * r1 * dx * (T2[1:n] - T1[1:n])) / (rho2 * Cp2 * dx * Ac2)
        dT2dt[0] = (m2 * Cp2 * (T2i - T2[0]) - U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho2 * Cp2 * dx * Ac2)
        return dT2dt
    
    T_out1 = odeint(dT1dt_function, T1, t)
    T_out1 = T_out1
    T_out2 = odeint(dT2dt_function, T2, t)
    T_out2 = T_out2
    
    # Criação dos DataFrames
    df_Temp1 = pd.DataFrame(np.array(T_out1), columns=x)
    df_Temp2 = pd.DataFrame(np.array(T_out2), columns=x)
    
    # Criando as paletas de cores para os fluidos 1 e 2
    paleta_calor = blend_palette(['blue', 'yellow', 'orange','red'], as_cmap=True, n_colors=100)
    
    # Função que atualiza o plot para o Fluido 1
    def update_plot1(t):
        plt.clf()
        line = pd.DataFrame(df_Temp1.iloc[t, :]).T
        sns.heatmap(line, cmap=paleta_calor)
        plt.title(f'Time: {t} (s)')
        plt.gca().set_xticklabels(['{:.2f}'.format(val) for val in x])
        
    # Função que atualiza o plot para o Fluido 2
    def update_plot2(t):
        plt.clf()
        line = pd.DataFrame(df_Temp2.iloc[t, :]).T
        sns.heatmap(line, cmap=paleta_calor)
        plt.title(f'Time: {t} (s)')
        plt.gca().set_xticklabels(['{:.2f}'.format(val) for val in x])

    # Criação e exibição da figura 1
    fig_ani1 = plt.figure(figsize=(8,6))
    ani1 = FuncAnimation(fig_ani1, update_plot1, frames=df_Temp1.shape[0], repeat=False)
    save1 = ani1.save('Temperature Variation - Fluid 1.gif', writer='pillow', fps=10)
    
    # Criação e exibição da figura 2
    fig_ani2 = plt.figure(figsize=(8,6))
    ani2 = FuncAnimation(fig_ani2, update_plot2, frames=df_Temp2.shape[0], repeat=False)
    save2 = ani2.save('Temperature Variation - Fluid 2.gif', writer='pillow', fps=10)
    
    # Exibindo a simulação
    with st.expander("Real-Time Simulation Visualization for Fluid 1 (cold) (Click here to view)"):
        st.write('Temperature variation of fluid 1 (cold) over time and along length.')
        st.write('Time is shown above the GIF in seconds. Temperatures (in Kelvin) are displayed on the variable y-axis scale. The length of the heat exchanger is represented in meters along the GIF’s x-axis.')
        st.image('Temperature Variation - Fluid 1.gif')
    with st.expander("Real-Time Simulation Visualization for Fluid 2 (warm) (Click here to view)"):
        st.write('Temperature variation of fluid 2 (warm) over time and along length.')
        st.write('Time is shown above the GIF in seconds. Temperatures (in Kelvin) are displayed on the variable y-axis scale. The length of the heat exchanger is represented in meters along the GIF’s x-axis.')
        st.image('Temperature Variation - Fluid 2.gif')
        
    # Exibindo o gráfico de variação da temperatura ao longo do comprimento em regime permanente para ambos os fluidos
    plt.figure(fig_permanente)
    plt.plot(x, df_Temp1.iloc[-1, :] , color='blue', label='Cold fluid')
    plt.plot(x, df_Temp2.iloc[-1, :], color='red', label='Warm Fluid')
    plt.xlabel('Length (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.title('Temperature of both cold and warm fluids along the heat exchanger length under steady-state conditions.')
    st.pyplot(plt)

st.title('Simulation Input Parameters')
# Valores input
L = st.number_input('Tube length (m)', min_value=0.0)
r1 = st.number_input('Inner radius of the tube (m)', min_value=0.0)
r2 = st.number_input('Outer radius of the tube (m)', min_value=0.0)
n = st.number_input('Number of nodes for discretization', min_value=1)
m1 = st.number_input('Mass flow of the cold fluid (kg/s)', min_value=0.0)
Cp1 = st.number_input('Specific heat capacity of the cold fluid (J/kg.K)', min_value=0.0)
rho1 = st.number_input('Specific mass of the cold fluid (kg/m³)', min_value=0.0)
m2 = st.number_input('Mass flow of the warm fluid (kg/s)', min_value=0.0)
Cp2 = st.number_input('Specific heat capacity of the warm fluid (J/kg.K)', min_value=0.0)
rho2 = st.number_input('Specific mass of the warm fluid (kg/m³)', min_value=0.0)
T1i = st.number_input('Inlet temperature of the cold fluid in the heat exchanger (K)')
T2i = st.number_input('Inlet temperature of the warm fluid in the heat exchanger (K)')
T0 = st.number_input('Initial temperature of the heat exchanger (K)')
U = st.number_input('Overall heat transfer coefficient (W/m².K)', min_value=0.0)
dx = L / n

t_final = st.number_input('Simulation Time (s)', min_value=0.0)
dt = st.number_input('Time Step (s)', min_value=0.0)

if st.button('Run Simulation'):
    run_simulation(L, r1, r2, n, m1, Cp1, rho1, m2, Cp2, rho2, T1i, T2i, T0, U, dx, t_final, dt)
elif st.button('Run Standard Example'):
    run_simulation(10, 0.1, 0.15, 10, 3, 4180, 995.61, 5, 4180, 995.61, 400, 800, 300, 1500, 10 / 10, 100, 1)
