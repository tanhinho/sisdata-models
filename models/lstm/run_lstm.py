from lstm import LSTMModel, LSTMOptimizer
from datasets import DatasetA


def run_lstm(seed):
    print("Optimizing parameters for LSTM model and dataset A...")
    df_train, df_val, df_test = DatasetA.load_and_preprocess()
    lstm_optimizer = LSTMOptimizer(
        df_train=df_train,
        df_val=df_val,
        input_size=len(DatasetA.FEATURE_COLS),
        seed=seed,
        feature_cols=DatasetA.FEATURE_COLS,
        target_col=DatasetA.TARGET_COL,
    )
    results = lstm_optimizer.optimize()
    print(f"Best parameters found: {results['best_params']}")
    print(f"Best fitness achieved: {results['best_fitness']}")
    print("Parameter optimization for LSTM model complete.\n")

    print("Training LSTM model with best parameters...")
    lstm_model = LSTMModel(
        input_size=len(DatasetA.FEATURE_COLS),
        output_size=1,
        hidden_size=results['best_params']['hidden_size'],
        num_layers=results['best_params']['num_layers'],
        dropout=results['best_params']['dropout'],
        feature_cols=DatasetA.FEATURE_COLS,
        target_col=DatasetA.TARGET_COL,
    )
    lstm_mse = lstm_model.fit_and_evaluate(
        df_train=df_train,
        df_test=df_test,
        seq_length=results['best_params']['sequence_length'],
        lr=results['best_params']['learning_rate'],
        epochs=results['best_params']['epochs'],
        batch_size=results['best_params']['batch_size'],
    )
    print(f"LSTM model trained. Test MSE: {lstm_mse}\n")
