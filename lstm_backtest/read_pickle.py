import pickle
import objects
def read_future_file():
    with open("/Users/ericervin/Documents/Coding/sigma/lstm_backtest/lstm_entries_1e5b9ae6/0.pkl", "rb") as f:
        data = pickle.load(f)
        print(data['recommendations'])
if __name__ == '__main__':
    read_future_file()