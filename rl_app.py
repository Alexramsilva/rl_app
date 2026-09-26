import streamlit as st
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt

from stable_baselines3 import PPO


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Nopal Quant - RL Trading",
    page_icon="📈",
    layout="wide"
)

st.image("NQH.svg", caption="", width=400)
st.title("")
st.subheader("Reinforcement Learning Trading Agent")

st.write(
    "Agente PPO para generar señales de compra y venta "
    "a partir de información histórica de Yahoo Finance."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Parámetros del modelo")

ticker = st.sidebar.text_input(
    "Clave de Yahoo Finance",
    value="BTC-USD"
).upper().strip()

periodo = st.sidebar.selectbox(
    "Periodo histórico",
    ["120d","6mo", "1y", "2y", "5y", "10y"],
    index=2
)

window = st.sidebar.slider(
    "Ventana de observaciones",
    min_value=5,
    max_value=30,
    value=10
)

cost = st.sidebar.number_input(
    "Costo de transacción",
    min_value=0.0,
    max_value=0.02,
    value=0.0025,
    step=0.0001,
    format="%.4f"
)

timesteps = st.sidebar.selectbox(
    "Pasos de entrenamiento PPO",
    [10_000, 25_000, 50_000, 100_000],
    index=3
)

entrenar = st.sidebar.button(
    "Entrenar agente PPO",
    type="primary"
)


# ============================================================
# ENTORNO DE TRADING
# ============================================================

class TradingEnv(gym.Env):

    def __init__(self, df, window=10, cost=0.001):

        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window = window
        self.cost = cost

        # 0 = mantener
        # 1 = comprar / posición larga
        # 2 = vender / cerrar posición

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window + 2,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.t = self.window
        self.position = 0

        return self._get_obs(), {}

    def _get_obs(self):

        returns = self.df["Return"].iloc[
            self.t - self.window:self.t
        ].values

        rsi = self.df["RSI"].iloc[self.t]

        ma = self.df["MA"].iloc[self.t]

        return np.concatenate(
            [returns, [rsi, ma]]
        ).astype(np.float32)

    def step(self, action):

        prev_position = self.position

        # ----------------------------
        # Acción del agente
        # ----------------------------

        if action == 1:

            self.position = 1

        elif action == 2:

            self.position = 0

        # ----------------------------
        # Retorno
        # ----------------------------

        ret = self.df["Return"].iloc[self.t]

        reward = self.position * ret

        # ----------------------------
        # Costo de transacción
        # ----------------------------

        reward -= self.cost * abs(
            self.position - prev_position
        )

        # ----------------------------
        # Avanzar
        # ----------------------------

        self.t += 1

        terminated = self.t >= len(self.df) - 1

        truncated = False

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {}
        )


# ============================================================
# DESCARGA DE DATOS
# ============================================================

@st.cache_data
def descargar_datos(ticker, periodo):

    df = yf.download(
        ticker,
        period=periodo,
        auto_adjust=True,
        progress=False,
        interval="1h"
    )

    if df.empty:
        return None

    # Manejo de MultiIndex de Yahoo Finance
    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    df["Close"] = df["Close"].squeeze()

    df["Volume"] = df["Volume"].squeeze()

    # ----------------------------
    # Variables
    # ----------------------------

    df["Return"] = df["Close"].pct_change()

    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    ).rsi()

    df["MA"] = df["Close"].rolling(20).mean()

    df = df.dropna()

    return df


# ============================================================
# OBTENER DATOS
# ============================================================

df = descargar_datos(
    ticker,
    periodo
)


if df is None:

    st.error(
        f"No se encontraron datos para {ticker}."
    )

    st.stop()


# ============================================================
# INFORMACIÓN DEL ACTIVO
# ============================================================

st.success(
    f"Datos descargados correctamente: {ticker}"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Precio actual",
    f"${df['Close'].iloc[-1]:,.2f}"
)

col2.metric(
    "Observaciones",
    f"{len(df):,}"
)

col3.metric(
    "RSI",
    f"{df['RSI'].iloc[-1]:.2f}"
)

col4.metric(
    "MA(20)",
    f"${df['MA'].iloc[-1]:,.2f}"
)


# ============================================================
# GRÁFICO DEL PRECIO
# ============================================================

st.subheader("Precio histórico")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    df.index,
    df["Close"],
    label="Precio"
)

ax.plot(
    df.index,
    df["MA"],
    linestyle="--",
    label="MA(20)"
)

ax.set_title(
    f"{ticker} - Precio histórico"
)

ax.set_xlabel("Fecha")
ax.set_ylabel("Precio")

ax.legend()

ax.grid(alpha=0.3)

st.pyplot(fig)


# ============================================================
# ENTRENAMIENTO
# ============================================================

if entrenar:

    st.subheader(
        "Entrenamiento del agente PPO"
    )

    with st.spinner(
        "Entrenando agente de Reinforcement Learning..."
    ):

        # Crear entorno
        env = TradingEnv(
            df,
            window=window,
            cost=cost
        )

        # Modelo PPO
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0
        )

        # Entrenamiento
        model.learn(
            total_timesteps=timesteps
        )

    st.success(
        f"Entrenamiento terminado: "
        f"{timesteps:,} pasos"
    )


    # ========================================================
    # SIMULACIÓN
    # ========================================================

    st.subheader(
        "Simulación de la estrategia"
    )

    obs, _ = env.reset()

    terminated = False
    truncated = False

    capital = 1.0

    capital_history = [capital]

    actions = []

    positions = []

    rewards = []


    while not (terminated or truncated):

        # ----------------------------
        # Predicción
        # ----------------------------

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        action = int(action)

        # Guardar acción
        actions.append(action)

        # ----------------------------
        # Ejecutar acción
        # ----------------------------

        obs, reward, terminated, truncated, _ = env.step(
            action
        )

        # Guardar recompensa
        rewards.append(reward)

        # Guardar posición
        positions.append(env.position)

        # ----------------------------
        # Capital
        # ----------------------------

        capital *= (1 + reward)

        capital_history.append(
            capital
        )


    # ========================================================
    # DATAFRAME DE RESULTADOS
    # ========================================================

    df_test = df.iloc[
        window:window + len(actions)
    ].copy()

    df_test["Action"] = actions

    df_test["Position"] = positions

    df_test["Reward"] = rewards


    # ========================================================
    # RESULTADOS
    # ========================================================

    retorno_rl = capital - 1

    buy_hold = (
        df_test["Close"].iloc[-1]
        /
        df_test["Close"].iloc[0]
        - 1
    )


    st.subheader(
        "Resultados"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Capital final",
        f"{capital:.4f}"
    )

    col2.metric(
        "Retorno PPO",
        f"{retorno_rl * 100:.2f}%"
    )

    col3.metric(
        "Buy & Hold",
        f"{buy_hold * 100:.2f}%"
    )


    # ========================================================
    # CAPITAL
    # ========================================================

    st.subheader(
        "Evolución del capital"
    )

    fig2, ax2 = plt.subplots(
        figsize=(12, 5)
    )

    ax2.plot(
        capital_history,
        label="PPO"
    )

    ax2.axhline(
        1,
        linestyle="--",
        label="Capital inicial"
    )

    ax2.set_title(
        f"{ticker} - Capital de la estrategia"
    )

    ax2.set_xlabel("Periodo")

    ax2.set_ylabel("Capital")

    ax2.legend()

    ax2.grid(alpha=0.3)

    st.pyplot(fig2)


    # ========================================================
    # SEÑALES DE COMPRA Y VENTA
    # ========================================================

    st.subheader(
        "Señales de compra y venta"
    )

    # Comprar
    buy_signals = df_test[
        df_test["Action"] == 1
    ]

    # Vender
    sell_signals = df_test[
        df_test["Action"] == 2
    ]


    fig3, ax3 = plt.subplots(
        figsize=(12, 6)
    )


    # ----------------------------
    # Precio
    # ----------------------------

    ax3.plot(
        df_test.index,
        df_test["Close"],
        label="Precio"
    )


    # ----------------------------
    # Media móvil
    # ----------------------------

    ax3.plot(
        df_test.index,
        df_test["MA"],
        linestyle="--",
        label="MA(20)"
    )


    # ----------------------------
    # Compras
    # ----------------------------

    ax3.scatter(
        buy_signals.index,
        buy_signals["Close"],
        marker="^",
        color="green",
        s=90,
        label="Compra"
    )


    # ----------------------------
    # Ventas
    # ----------------------------

    ax3.scatter(
        sell_signals.index,
        sell_signals["Close"],
        marker="v",
        color="red",
        s=90,
        label="Venta"
    )


    ax3.set_title(
        f"{ticker} - Señales del agente PPO"
    )

    ax3.set_xlabel(
        "Fecha"
    )

    ax3.set_ylabel(
        "Precio"
    )

    ax3.legend()

    ax3.grid(alpha=0.3)

    st.pyplot(fig3)


    # ========================================================
    # TABLA DE OPERACIONES
    # ========================================================

    st.subheader(
        "Últimas decisiones del agente"
    )

    tabla = df_test[
        ["Close", "RSI", "MA", "Action", "Position", "Reward"]
    ].tail(20).copy()


    tabla["Decision"] = tabla["Action"].map({

        0: "Mantener",

        1: "Comprar",

        2: "Vender"

    })


    st.dataframe(
        tabla,
        use_container_width=True
    )


    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    compras = (
        df_test["Action"] == 1
    ).sum()

    ventas = (
        df_test["Action"] == 2
    ).sum()


    col1, col2 = st.columns(2)

    col1.metric(
        "Señales de compra",
        int(compras)
    )

    col2.metric(
        "Señales de venta",
        int(ventas)
    )


    # ========================================================
    # COMPARACIÓN
    # ========================================================

    st.subheader(
        "Comparación de estrategias"
    )


    if retorno_rl > buy_hold:

        st.success(
            "El agente PPO superó a Buy & Hold "
            "durante el periodo analizado."
        )

    else:

        st.warning(
            "Buy & Hold superó al agente PPO "
            "durante el periodo analizado."
        )


    st.info(
        "Los resultados corresponden a una simulación "
        "histórica y no garantizan rendimientos futuros."
    )

# ESTILO
# -----------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #5BF58E;
    }
    h1, h2, h3 {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)
