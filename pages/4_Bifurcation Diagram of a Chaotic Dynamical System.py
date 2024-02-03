import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.animation as animation
import streamlit.components.v1 as components
import os

st.title('Bifurcation Diagram of a Chaotic Dynamical System')

st.write('''
         A chaotic dynamical system is highly sensitive to initial conditions where small
         perturbations in the initial conditions can lead to drastically different outcomes, 
         so trajectories tend to be complex, unpredictable, and visually interesting!
         
         These systems are often described by a set of non-linear differential equations, and examples
         can be found in biology, economics, etc.
         
         In this example, let us explore how a chaotic system arises from a simple nonlinear system: the logistic map.  
         The logistic map models the evolution of a population, taking into account reproduction and density-dependent mortality (starvation).
         '''
)

col1, col2 = st.columns(2)
col1.header('The Logistic Function')
col1.write('''
         The logistic map is given by the following logistic function:
         ''')
col1.latex(r'''f_r(x) = r x (1 - x)''')
col1.write('''  
         In application, $x$ is the population and $r$ is the growth rate parameter.
         ''')
def logistic(r, x):
    return r * x * (1 - x)

x = np.linspace(0, 1)
fig, ax = plt.subplots(1, 1)
plt.plot(x, logistic(2, x), 'k')
col2.pyplot(fig)


st.header('Dynamics of the Logistic Map')
st.write('''
         The dynamics of the logistic map is defined by the recurrence relation:
            $x_{n+1}^{(r)} = f_r (x_{n}^{(r)}) = r x_{n}^{(r)} (1 - x_{n}^{(r)})$
        Once again, in application, $x_n$ is the population at time $n$, and $r$ is the growth rate parameter.
            ''')

st.write('''
         Let's explore the behavior of the logistic map as we vary the growth rate parameter $r$. 
         To show how the population evolves over the time, we will plot points for $x_n$ featuring increasing opacity!
         I've shown two examples on the left. Adjust the parameters to view changes on the right plot.
         ''')

r = st.slider('r', min_value=0.0, max_value=4.0, value = 3.0)
n = st.slider('n', min_value=1, max_value=10, value=10)
on = st.toggle('Plot x = y line', False)


def plot_system(r, x0, n, ax=None):
    # Plot the function and the
    # y=x diagonal line.
    t = np.linspace(0, 1)
    ax.plot(t, logistic(r, t), 'k', lw=2)
    if on:
        ax.plot([0, 1], [0, 1], 'k', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    x = x0
    for i in range(n):
        y = logistic(r, x)
        # Plot the two lines.
        if on:
            ax.plot([x, x], [x, y], 'k', lw=1)
            ax.plot([x, y], [y, y], 'k', lw=1)
        # Plot the positions with increasing
        # opacity.
        ax.plot([x], [y], 'ok', ms=10,
                alpha=(i + 1) / n)
        x = y

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('r')
    ax.set_ylabel('$x^{(r)}$')
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6),
                               sharey=True)

plot_system(2.5, .1, 10, ax=ax1)
plot_system(3.5, .1, 10, ax=ax2)
plot_system(r, .1, n, ax=ax3)

st.pyplot(fig2)

st.write('''
         For $r=2.5$, the population converges! However for $r=3.5$, the population oscillates between two values. 
         ''')


st.header('Bifurcation Visualization for Different Values of r')
st.write(''' 
         Let us understand this behavior by plotting 10,000 values of **r** between 0 and 4.0 for 1000 iterations.  
         For avoid dense visualization, we will plot the last 100 iterations for each value of **r**.  
         We initialize the population at $x_0 = 0.00001$.
         ''')
n = 10000-1
r = np.linspace(0.0004, 4.0, n)

iterations = 1000
last = 100
x = 1e-5 * np.ones(n)
lyapunov = np.zeros(n)


st.write('''
         Let's also include an estimate of the Lyapunov exponent for the logistic map to characterize the model's sensitivity to initial conditions. Lyapunov exponent is a measure of the rate of separation of two infinitesimally close trajectories in a chaotic system. It is positive for a chaotic system, zero for a non-chaotic system, and negative for a stable system.
         It is approximated by: ''')
st.latex(r'''\lambda = \lim_{{n \to \infty}} \frac{1}{n} \sum_{{i=0}}^{{n-1}} \log \left| f'(x_i) \right|''' )


fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                               sharex=True)
for i in range(iterations):
    x = logistic(r, x)
    # We compute the partial sum of the
    # Lyapunov exponent.
    epsilon = 1e-10  # or any small non-zero value
    condition = abs(r - 2 * r * x) != 0
    lyapunov += np.where(condition, np.log(abs(r - 2 * r * x)), np.log(epsilon))
    
    # lyapunov += np.log(abs(r - 2 * r * x))

    # We display the bifurcation diagram.
    if i >= (iterations - last):
        ax1.plot(r, x, ',k', alpha=.25)
ax1.set_xlim(0, 4)
ax1.set_xlabel('r')
ax1.set_ylabel('$x^{(r)}_{n}$')
ax1.set_title("Bifurcation Diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.5)
# Negative Lyapunov exponent.
ax2.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0] / iterations,
         '.k', alpha=.5, ms=.5)
# Positive Lyapunov exponent.
ax2.plot(r[lyapunov >= 0],
         lyapunov[lyapunov >= 0] / iterations,
         '.r', alpha=.5, ms=.5)
ax2.set_xlim(0, 4)
ax2.set_ylim(-2, 1)
ax2.set_xlabel('r')
ax2.set_title("Lyapunov exponent")
plt.tight_layout()


st.pyplot(fig3)

st.write('''
         We observe chaotic behavior with bifurcations (Def: the division into two branches or parts.) !!  
         Notably, we observe:  
         - Single fixed point for $r<3$  
         - With r between 0 and 1, the population eventually dies, independent of the initial population  
         - Chaotic behavior for $r>3$  
         - With r between 3 and 1 + √6 ≈ 3.44949, the population approaches permanent oscillations between two values   
         - With r between 3.44949 and 3.54409, from almost all initial conditions the population will approach permanent oscillations among four values   
         - Finally, we observe the property of the Lyapunov exponent: it is positive for r>3!
            ''')

# st.header('Animated Bifurcation Diagram')


# r = np.linspace(0.0, 4.0, n)
# iterations = 50

# fig4 = plt.figure(figsize=(5, 5))
# ax = plt.axes()

# def animate(i):
#     # clear axes object
#     plt.cla()
#     ax.set_title(f"Bifurcation Diagram for $x^{{(r)}}_{{n + 1}} = rx^{{(r)}}_{{n}}(1 - x^{{(r)}}_{{n}})$ for $n=${i + 1}")
#     ax.set_xlabel('r')
#     ax.set_ylabel('$x^{(r)}_{n}$')
#     ax.tick_params(axis='both', which='major')
#     ax.set_xlim(0.0, 4.0)
#     ax.set_ylim(0.0, 1.0)
#     # plt.gcf().text(0.13, 0.15, 'by Vladimir Ilievski', fontfamily='Verdana')
    
#     x = np.random.uniform(0.0, 1.0, n)  # initial value
#     for _ in range(i + 1):  # animated iterations
#         x = logistic(r, x)
    
#     l = ax.plot(r, x, ',b', alpha=.5)
#     return l

# # call the animator	 
# anim = animation.FuncAnimation(fig4, animate, frames=iterations, interval=225, blit=True)

# path = "./bifurcation.html"

    
# if os.path.exists(path):
#     HtmlFile = open(path, "r")
#     source_code = HtmlFile.read() 
#     components.html(source_code, height = 900,width=900)
# else:
#     with open(path,"w") as f:
#         print(anim.to_html5_video(), file=f)
        
#     HtmlFile = open(path, "r")
#     source_code = HtmlFile.read() 
#     components.html(source_code, height = 900,width=900)
  
