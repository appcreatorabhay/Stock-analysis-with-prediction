import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set up the Streamlit app title and custom styles with a dark background
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #121212;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    h1, h2 {
        color: #ff6347;
    }
    .stTextInput input {
        background-color: #333333;
        color: white;
    }
    .stSlider .st-bd {
        color: white;
    }
    .stButton button {
        background-color: #ff6347;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    .stDataFrame, .stTable {
        background-color: #222222;
        color: white;
        border: 1px solid #444444;
    }
    </style>
    """, unsafe_allow_html=True
)

# Define the start and end dates
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set up the Streamlit app title
st.title('Stock Forecast App')

# Create a dictionary to map company names to ticker symbols
company_ticker_map = {
    'Google': 'GOOG',
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'GameStop': 'GME',
    'Amazon': 'AMZN',
    'Tesla': 'TSLA',
    'Meta': 'META',
    'Nvidia': 'NVDA',
    'Netflix': 'NFLX',
    'Adobe': 'ADBE'
}

# Create a text input for the user to type the company name
company_name = st.text_input('Enter company name (e.g., Google, Apple, Microsoft):')

# If the entered company name is valid, map it to its ticker symbol
if company_name in company_ticker_map:
    selected_stock = company_ticker_map[company_name]
else:
    selected_stock = None

# Check if the company name exists in the dictionary, otherwise display an error
if selected_stock is None and company_name != "":
    st.error(f"Company '{company_name}' not found. Please check the spelling or use a valid company name.")

# If a valid company ticker is selected, proceed with prediction
if selected_stock:
    # Create an input box for the user to specify the number of years for prediction
    years_input = st.text_input('Enter number of years for prediction (e.g., 1, 2, 3):', '1')

    # Validate the input to ensure it's a valid number
    try:
        n_years = int(years_input)
        if n_years < 1 or n_years > 4:
            st.error("Please enter a number between 1 and 4.")
        else:
            period = n_years * 365  # Number of days in the forecast period
    except ValueError:
        st.error("Please enter a valid number for years.")

    if 'n_years' in locals() and n_years >= 1 and n_years <= 4:
        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw Data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
            fig.layout.update(title_text='Time Series Data with Range Slider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast Data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast Components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
