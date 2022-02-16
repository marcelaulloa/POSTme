from fast_ml.model_development import train_valid_test_split
import matplotlib.pyplot as plt


# FUNCTION --> # Shows you a report with the main info of each column from data frame
def df_main_insights(data_frame):
    print("\nThis data frame have the following shape: ", data_frame.shape)
    for col_name in data_frame.columns:
        print(
            "\nThe column:",
            '"',
            col_name,
            '"',
            "corresponds to index:",
            '"',
            data_frame.columns.get_loc(col_name),
            '"',
            "contains this amount of unique values inside:",
            '"',
            len(data_frame[col_name].unique()),
            '"',
            "contain this type of data:",
            data_frame[col_name].dtypes,
            '"',
        )

# FUNCTION --> # Identifies all unique values for all df columns
def df_identification_of_unique_values_per_column(data_frame):
    # variable to hold the count
    cnt = 0
    # list to hold visited values
    visited = []
    for col_name in data_frame.columns:
        # Listing all unique values that exist in the specified colum of the data frame
        print(
            "\nColumn name:",
            col_name,
            "have these unique data elements:\n\n",
            '"',
            data_frame[col_name].unique(),
            '"',
        )

# FUNCTION --> # Shows you (if applies) in which colums you already have NaN and/or Null values
def df_check_of_NaNs_and_Nulls(data_frame):
    print(
        "\nYour dataframe NaN content is:\t\n\n", data_frame.isna().values.any(), "\n"
    )
    print(
        "\nYour dataframe Null content is:\t\n\n",
        data_frame.isnull().values.any(),
        "\n",
    )

# FUNCTION --> Model Evaluation Function based on SCORES
model_accuracy = {}  # Empty dictionary for holding the model accuracy


def model_evaluation(model_name, predictions):

    # Evaluation Summary:
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("----------------------------------------------------------")
    print(
        "Standard Confussion Matrix (error matrix):\n",
        confusion_matrix(y_test, predictions),
    )
    # Uncomment here those Metrics you want to use (leave as comment those not needed)
    f1_macro = f1_score(y_test, predictions, average="macro")
    f1_micro = f1_score(y_test, predictions, average="micro")
    f1_weighted = f1_score(y_test, predictions, average="weighted")
    f1 = np.min(f1_score(y_test, predictions, average=None))
    # recall = accuracy_score(y_test, predictions)
    # precision = accuracy_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy Score obtained is: %.2f%%" % (accuracy * 100.0))
    print("----------------------------------------------------------")
    print("f1_macro Score obtained is: %.2f%%" % (f1_macro * 100.0))
    print("----------------------------------------------------------")
    print("f1_micro Score obtained is: %.2f%%" % (f1_micro * 100.0))
    print("----------------------------------------------------------")
    print("f1_weighted Score obtained is: %.2f%%" % (f1_weighted * 100.0))
    print("----------------------------------------------------------")
    print("f1 Score obtained is: %.2f%%" % (f1 * 100.0))

    model_accuracy[f"{model_name}"] = accuracy

def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def plot_metrics(history):

  keys = history.history.keys()
  metrics = ['mse', 'mae', 'r_square']
  plt.figure(figsize=(12, 10))
  for n, metric in enumerate(metrics):
    name = metric.capitalize()
    metric_name = [item for item in keys if metric in item and 'val_'+metric not in item][0]
    val_metric_name = [item for item in keys if 'val_'+metric in item][0]
    
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric_name], color='b', label='Train')
    plt.plot(history.epoch, history.history[val_metric_name], color='r', 
             linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()

# Train Test Val Split Function

def train_test_variables(df_model,target,drop):

  df_model = df_model.drop(columns=[drop])

  X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split(df_model, target = target,train_size=0.8, valid_size=0.1, test_size=0.1)

  return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data(X_train,X_val,X_test):
  
  train_clean_text = X_train['processed_text']
  val_clean_text = X_val['processed_text']
  test_clean_text = X_test['processed_text']

  X_train = X_train.drop(columns='processed_text')
  X_val = X_val.drop(columns='processed_text')
  X_test = X_test.drop(columns='processed_text')

  return X_train, X_val, X_test, train_clean_text, val_clean_text, test_clean_text


from keras import backend as K

def transformer():
  categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
  )

  numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
  return categorical_transformer, numeric_transformer

def regr_report(y_test, y_pred, X_test):

  print('MAE', metrics.mean_absolute_error(y_test, y_pred))
  print('MSE', metrics.mean_squared_error(y_test, y_pred))
  print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('R2 Score', metrics.r2_score(y_test, y_pred))

  # precidtion

  predictions = regr.predict(X_test)

  plt.scatter(y_test, predictions, color="red")
  plt.xlabel("True Values")
  plt.ylabel("Predictions");

def print_regression_metrics(history):
    """
    Print MSE and R2 metrics for training and validation set.
    """
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    val_r2 = history.history['val_r_square']
    r2 = history.history['r_square']
    i = np.argmin(val_loss)
    print("loss: {}, r_square: {}, val_loss: {}, val_r_square: {}".format(loss[i], r_square[i], val_loss[i], val_r_square[i]))
    return

def time_day(x):
  if x > 4 and x <=12:
    return 'morning'
  elif x > 12 and x <=18:
    return 'afternoon'
  else:
    return 'evening'