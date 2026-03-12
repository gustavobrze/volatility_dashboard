import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Dashboard Risco", layout="wide")
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=1800)
def get_data(tickers, benchmarks, start_date, end_date):
    
    all_tickers = tickers + benchmarks

    df = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError("Sem dados informados.")

    try:
        data = df['Close']

    except KeyError:

        st.error("Erro. Verifique a estrutura de dados.")
        st.write(df.head())
        st.stop()

    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = all_tickers

    data = data.dropna(axis=1, how='all')
    
    data = data.ffill().dropna()
    
    return data

def calculate_log_returns(prices):
    
    return np.log(prices / prices.shift(1)).dropna()

def calculate_ewma_volatility(returns, lambda_param):

    vol_daily = returns.ewm(alpha=1-lambda_param, adjust=False).std()
    return vol_daily * np.sqrt(252)

def calculate_portfolio_vol_series(returns, weights, lambda_param):
    
    cov_matrix_series = returns.ewm(alpha=1-lambda_param, adjust=False).cov()
    
    port_vols = []
    dates = returns.index
    
    w = np.array(weights)
    
    for d in dates:
        try:
            
            cov_t = cov_matrix_series.loc[d].values
            
            port_var = np.dot(w.T, np.dot(cov_t, w))
            port_vols.append(np.sqrt(port_var))

        except KeyError:
            port_vols.append(np.nan)
            
    return pd.Series(port_vols, index=dates) * np.sqrt(252)

st.title("Volatilidade e Correlação")

with st.sidebar:
    st.header("Parâmetros do Portfólio")
    
    default_tickers = "MELI, UNH, PANW, ORCL, PLTR, JPM, AMZN, POMO4.SA, HASH11.SA, SOLH11.SA, BRAV3.SA, BOVA11.SA"
    ticker_input = st.text_input("Ativos (separados por vírgula)", default_tickers)
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    
    # Weight Input
    st.write("---")
    weight_mode = st.radio("Weighting Scheme", ["Equal Weights", "Custom (Long/Short)"])
    
    if weight_mode == "Custom (Long/Short)":
        st.caption("Use negative numbers for short positions (e.g., 0.5, -0.2, 0.3)")
        weight_input = st.text_input("Weights (comma separated)", 
                                     ", ".join([str(round(1/len(tickers), 2))]*len(tickers)))
        try:
            weights = [float(w) for w in weight_input.split(',')]
            if len(weights) != len(tickers):
                st.error("Number of weights must match number of assets.")
                st.stop()
            
            weights = np.array(weights)
            
            # Optional: Normalize by Gross Exposure so the total book size = 100%
            # gross_exposure = np.sum(np.abs(weights))
            # if gross_exposure > 0:
            #     weights = weights / gross_exposure
                
        except ValueError:
            st.error("Invalid weight format. Please use numbers only.")
            st.stop()
    else:
        # Equal weights (Long only)
        weights = np.array([1/len(tickers)] * len(tickers))
        
    st.info(f"Active Weights: {weights}")
    
    # Calculate Net and Gross Exposure for the analyst's awareness
    net_exposure = np.sum(weights)
    gross_exposure = np.sum(np.abs(weights))
    st.caption(f"**Net Exposure:** {net_exposure:.0%} | **Gross Exposure:** {gross_exposure:.0%}")

    st.write("---")
    st.header("Parâmetros de Cálculo")
    lambda_param = st.slider("Fator de decaimento (λ)", 0.80, 0.99, 0.94, 0.01, 
                             help="λ maior = Decaimento mais lento (suavizado). λ menor = Reação mais rápida. 0.94 é o valor padrão.")
    
    lookback_years = st.slider("Anos de Histórico", 1, 10, 2)
    start_date = datetime.today() - timedelta(days=lookback_years*365)
    end_date = datetime.today()

if len(tickers) > 0:
    with st.spinner('Obtendo dados e calculando métricas...'):
        
        benchmarks = ['^GSPC', '^BVSP']
        try:
            prices = get_data(tickers, benchmarks, start_date, end_date)
        except Exception as e:
            st.error(f"Erro obtendo dados: {e}")
            st.stop()
            
        asset_prices = prices[tickers]
        benchmark_prices = prices[benchmarks]
        
        log_returns = calculate_log_returns(asset_prices)
        bench_returns = calculate_log_returns(benchmark_prices)
        
        port_vol_ts = calculate_portfolio_vol_series(log_returns, weights, lambda_param)
        
        asset_vols = calculate_ewma_volatility(log_returns, lambda_param)
        
        bench_vols = calculate_ewma_volatility(bench_returns, lambda_param)
        
    tab_vol, tab_corr = st.tabs(["📉 Monitor de volatilidade", "🔗 Monitor de correlação"])

    with tab_vol:
        col1, col2, col3 = st.columns(3)
        current_vol = port_vol_ts.iloc[-1]
        
        with col1:
            st.metric("Volatilidade anualizada do portfólio", f"{current_vol:.2%}", 
                      delta=f"{(current_vol - port_vol_ts.iloc[-5]):.2%} (Var. 5d)", delta_color="inverse")
        with col2:
            sp500_vol = bench_vols['^GSPC'].iloc[-1]
            st.metric("Volatilidade S&P 500", f"{sp500_vol:.2%}")
        with col3:
            ibov_vol = bench_vols['^BVSP'].iloc[-1]
            st.metric("Volatilidade Ibovespa", f"{ibov_vol:.2%}")

        st.subheader("Volatilidade total do portfólio vs Benchmarks")
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=port_vol_ts.index, y=port_vol_ts, name='Portfólio', line=dict(color='cyan', width=3)))
        fig_vol.add_trace(go.Scatter(x=bench_vols.index, y=bench_vols['^GSPC'], name='S&P 500', line=dict(dash='dot', color='gray')))
        fig_vol.add_trace(go.Scatter(x=bench_vols.index, y=bench_vols['^BVSP'], name='Ibovespa', line=dict(dash='dot', color='green')))
        fig_vol.update_layout(yaxis_tickformat='.0%', template="plotly_dark", height=500, title=f"Volatilidade EWMA (λ={lambda_param})")
        st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Volatilidade individual dos ativos")
        
        fig_ind = px.line(asset_vols, title="Volatilidade individual anualizada dos ativos")
        fig_ind.update_layout(yaxis_tickformat='.0%', template="plotly_dark")
        st.plotly_chart(fig_ind, use_container_width=True)

    with tab_corr:
        st.subheader("Análise de correlação")

        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.markdown("#### Matriz de correlação dos ativos")
            
            ewma_cov = log_returns.ewm(alpha=1-lambda_param).cov()
            
            last_date_idx = ewma_cov.index.get_level_values(0)[-1]
            current_cov = ewma_cov.loc[last_date_idx]
            
            cov_values = current_cov.values
            
            inv_vols = 1.0 / np.sqrt(np.diag(cov_values))
            
            D_inv = np.diag(inv_vols)
            
            corr_values = D_inv @ cov_values @ D_inv
            
            current_corr_df = pd.DataFrame(corr_values, index=tickers, columns=tickers)
            
            np.fill_diagonal(current_corr_df.values, 1.0)

            fig_heat = px.imshow(current_corr_df, 
                                 text_auto='.2f', 
                                 color_continuous_scale='RdBu_r', 
                                 zmin=-1, zmax=1,
                                 aspect="auto")
                                 
            fig_heat.update_layout(
                template="plotly_dark", 
                title=f"Correlação ({last_date_idx.date()})",
                height=450
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
            with st.expander("Ver dados brutos de correlação"):
                st.dataframe(current_corr_df.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))

        with col_c2:
            st.markdown("#### Correlação do portfólio vs Benchmarks")
            
            port_daily_ret = (log_returns * weights).sum(axis=1)
            
            roll_window = 60
            corr_sp500 = port_daily_ret.rolling(roll_window).corr(bench_returns['^GSPC'])
            corr_ibov = port_daily_ret.rolling(roll_window).corr(bench_returns['^BVSP'])
            
            df_corr_bench = pd.DataFrame({
                "vs S&P 500": corr_sp500,
                "vs IBOVESPA": corr_ibov
            })
            
            fig_bench_corr = px.line(df_corr_bench, title=f"Correlação móvel de {roll_window} dias do portfólio vs Benchmarks")
            fig_bench_corr.update_layout(template="plotly_dark", yaxis_range=[-1, 1])
            
            fig_bench_corr.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_bench_corr, use_container_width=True)

else:
    st.warning("Insira tickers válidos.")