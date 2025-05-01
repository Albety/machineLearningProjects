import ccxt  # Library for connecting to crypto exchanges
import pandas as pd  # For data processing
import numpy as np  # For numerical operations
import ta  # Technical indicators
import time  # To control API request frequency
import logging  # For error handling
import tensorflow as tf  # AI for price prediction
from sklearn.preprocessing import MinMaxScaler  # Normalizing data

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Initialize Binance API
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

# Trading parameters
symbol = 'BTC/USDT'  # Target trading pair
leverage = 5  # Risky, but high potential return
risk_per_trade = 0.02  # 2% of portfolio risk per trade

# Strategy settings
STOP_LOSS_PERCENT = 0.015  # 1.5% stop-loss
TAKE_PROFIT_MULTIPLIER = 2  # Risk-to-reward ratio of 1:2

scaler = MinMaxScaler()


def fetch_data(symbol, timeframe='5m', limit=100):
    """Fetch historical OHLCV data"""
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def calculate_indicators(df):
    """Compute technical indicators for signal generation"""
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['supertrend'] = ta.trend.STCIndicator(df['close']).stc()
    return df


def generate_trade_signal(df):
    """Generate buy or sell signals"""
    latest = df.iloc[-1]
    if latest['macd'] > latest['macd_signal'] and latest['rsi'] < 70 and latest['supertrend'] > 50:
        return 'BUY'
    elif latest['macd'] < latest['macd_signal'] and latest['rsi'] > 30 and latest['supertrend'] < 50:
        return 'SELL'
    return None


def execute_trade(order_type, amount):
    """Execute a trade on Binance"""
    try:
        if order_type == 'BUY':
            order = exchange.create_market_buy_order(symbol, amount)
        elif order_type == 'SELL':
            order = exchange.create_market_sell_order(symbol, amount)
        logging.info(f"Trade executed: {order}")
    except Exception as e:
        logging.error(f"Error executing trade: {e}")


def train_ai_model(df):
    """Train an AI model to predict future prices"""
    df_scaled = scaler.fit_transform(df[['close']])
    X, y = [], []
    for i in range(50, len(df_scaled)):
        X.append(df_scaled[i - 50:i, 0])
        y.append(df_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16)
    return model


def predict_price(model, df):
    """Use AI model to predict the next closing price"""
    df_scaled = scaler.transform(df[['close']].tail(50))
    X = np.array([df_scaled[:, 0]])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    predicted_price = scaler.inverse_transform(model.predict(X))
    return predicted_price[0, 0]


def backtest_strategy():
    """Backtest the strategy using historical data"""
    df = fetch_data(symbol, '5m', 500)
    df = calculate_indicators(df)
    df['signal'] = df.apply(lambda row: generate_trade_signal(df), axis=1)
    model = train_ai_model(df)
    df['ai_predicted_price'] = df.apply(lambda row: predict_price(model, df), axis=1)

    capital = 1000  # Starting capital
    position = 0  # Track open positions
    for i in range(len(df)):
        if df['signal'][i] == 'BUY' and position == 0:
            position = capital / df['close'][i]
            capital -= position * df['close'][i]
        elif df['signal'][i] == 'SELL' and position > 0:
            capital += position * df['close'][i]
            position = 0
    return capital


def main():
    model = train_ai_model(fetch_data(symbol, '5m', 500))
    while True:
        df = fetch_data(symbol)
        df = calculate_indicators(df)
        signal = generate_trade_signal(df)
        ai_price = predict_price(model, df)
        if signal and (ai_price > df['close'].iloc[-1] if signal == 'BUY' else ai_price < df['close'].iloc[-1]):
            execute_trade(signal, 0.01)  # Example: Buy/Sell 0.01 BTC
        time.sleep(300)  # Wait 5 minutes before checking again


if __name__ == '__main__':
    main()
