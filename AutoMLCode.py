import sys
import os
sys.path.insert( 0, os.path.abspath("../../common") )
import json
import util
import boto3
import s3fs
import pandas as pd
import dateutil
import datetime

# **************** Create an instance of AWS SDK client for Amazon Forecast *************
today = datetime.datetime.now()
date_time = today.strftime("%m/%d/%Y, %H:%M:%S")
region = 'us-east-1'
session = boto3.Session(region_name=region)
forecast = session.client(service_name='forecast')
#forecastquery = session.client(service_name='forecastquery')

# Checking to make sure we can communicate with Amazon Forecast
assert forecast.list_predictors()
print(forecast.list_predictors())
#***************** Setup IAM Role used by Amazon Forecast to access your data ****************************
role_name = "automl-forecast-role"
print(f"Creating Role {role_name}...")
role_arn = util.get_or_create_iam_role( role_name = role_name )

# echo user inputs without account
print(f"Success! Created role = {role_arn.split('/')[1]}")
#*************** Import your data. ****************

key="C:/Temp/PythonProjects/auotml_stuff-main/Datasets/SeasonalData_1Y.csv"

automl_df = pd.read_csv(key, dtype = object, names=['timestamp','item_id','target_value'])

#display(automl_df.head(3))

bucket_name = "automlforecast"
print(f"\nAttempting to upload the data to the S3 bucket '{bucket_name}' at key '{key}' ...")

s3 = boto3.Session().resource('s3')
bucket = s3.Bucket(bucket_name)
if not bucket.creation_date:
    if region != "us-east-1":
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
    else:
        s3.create_bucket(Bucket=bucket_name)

s3.Bucket(bucket_name).Object(key).upload_file(key)
ts_s3_path = f"s3://{bucket_name}/{key}"

print(f"\nDone, the dataset is uploaded to S3 at {ts_s3_path}.")
# *********************** Creating the Dataset ********************
DATASET_FREQUENCY = "D" # H for hourly.
TS_DATASET_NAME = "AUTOML4"
#index,instance,Jobname,runtime,date
TS_SCHEMA = {
   "Attributes":[
      {
         "AttributeName":"timestamp",
         "AttributeType":"timestamp"
      },
      {
         "AttributeName":"item_id",
         "AttributeType":"string"
      },
      {
         "AttributeName":"target_value",
         "AttributeType":"integer"
      }
   ]
}

print('Creating dataset response...')
create_dataset_response = forecast.create_dataset(Domain="CUSTOM",
                                                  DatasetType='TARGET_TIME_SERIES',
                                                  DatasetName=TS_DATASET_NAME,
                                                  DataFrequency=DATASET_FREQUENCY,
                                                  Schema=TS_SCHEMA)
print('Creating dataset arn...')
ts_dataset_arn = create_dataset_response['DatasetArn']
describe_dataset_response = forecast.describe_dataset(DatasetArn=ts_dataset_arn)

print(f"The Dataset with ARN {ts_dataset_arn} is now {describe_dataset_response['Status']}.")

#-***********************Importing the Dataset ************
print('*********************** Importing the Dataset ************')
TIMESTAMP_FORMAT = "yyyy-MM-dd hh:mm:ss"
TS_IMPORT_JOB_NAME = "AUTOML4"
TIMEZONE = "EST"

ts_dataset_import_job_response = \
    forecast.create_dataset_import_job(DatasetImportJobName=TS_IMPORT_JOB_NAME,
                                       DatasetArn=ts_dataset_arn,
                                       DataSource= {
                                         "S3Config" : {
                                             "Path": ts_s3_path,
                                             "RoleArn": role_arn
                                         }
                                       },
                                       TimestampFormat=TIMESTAMP_FORMAT,
                                       TimeZone = TIMEZONE)

ts_dataset_import_job_arn = ts_dataset_import_job_response['DatasetImportJobArn']
describe_dataset_import_job_response = forecast.describe_dataset_import_job(DatasetImportJobArn=ts_dataset_import_job_arn)

print(f"Waiting for Dataset Import Job with ARN {ts_dataset_import_job_arn} to become ACTIVE. This process could take 5-10 minutes.\n\nCurrent Status:")

status = util.wait(lambda: forecast.describe_dataset_import_job(DatasetImportJobArn=ts_dataset_import_job_arn))

describe_dataset_import_job_response = forecast.describe_dataset_import_job(DatasetImportJobArn=ts_dataset_import_job_arn)
print(f"\n\nThe Dataset Import Job with ARN {ts_dataset_import_job_arn} is now {describe_dataset_import_job_response['Status']}.")

#**************** Creating a DatasetGroup ***************
print ('**************** Creating a DatasetGroup ***************')
DATASET_GROUP_NAME = "AUTOML4"
DATASET_ARNS = [ts_dataset_arn]

create_dataset_group_response = \
    forecast.create_dataset_group(Domain="CUSTOM",
                                  DatasetGroupName=DATASET_GROUP_NAME,
                                  DatasetArns=DATASET_ARNS)

dataset_group_arn = create_dataset_group_response['DatasetGroupArn']
describe_dataset_group_response = forecast.describe_dataset_group(DatasetGroupArn=dataset_group_arn)

print(f"The DatasetGroup with ARN {dataset_group_arn} is now {describe_dataset_group_response['Status']}.")

#***********************************************************************************

#*-******************Train a predictor************************
print('******************Train a predictor************************')
PREDICTOR_NAME = "AUTOML4"
FORECAST_HORIZON = 24
FORECAST_FREQUENCY = "D"
HOLIDAY_DATASET = [{
        'Name': 'holiday',
        'Configuration': {
        'CountryCode': ['US']
    }
}]

create_auto_predictor_response = \
    forecast.create_auto_predictor(PredictorName = PREDICTOR_NAME,
                                   ForecastHorizon = FORECAST_HORIZON,
                                   ForecastFrequency = FORECAST_FREQUENCY,
                                   DataConfig = {
                                       'DatasetGroupArn': dataset_group_arn,
                                       'AdditionalDatasets': HOLIDAY_DATASET
                                    },
                                   ExplainPredictor = True)

predictor_arn = create_auto_predictor_response['PredictorArn']
print(f"Waiting for Predictor with ARN {predictor_arn} to become ACTIVE. Depending on data size and predictor setting，it can take several hours to be ACTIVE.\n\nCurrent Status:")

status = util.wait(lambda: forecast.describe_auto_predictor(PredictorArn=predictor_arn))

describe_auto_predictor_response = forecast.describe_auto_predictor(PredictorArn=predictor_arn)
print(f"\n\nThe Predictor with ARN {predictor_arn} is now {describe_auto_predictor_response['Status']}.")

#********* Review accuracy metrics ************
print('********* Review accuracy metrics ************')
get_accuracy_metrics_response = forecast.get_accuracy_metrics(PredictorArn=predictor_arn)
wql = get_accuracy_metrics_response['PredictorEvaluationResults'][0]['TestWindows'][0]['Metrics']['WeightedQuantileLosses']
accuracy_scores = get_accuracy_metrics_response['PredictorEvaluationResults'][0]['TestWindows'][0]['Metrics']['ErrorMetrics'][0]

print(f"Weighted Quantile Loss (wQL): {json.dumps(wql, indent=2)}\n\n")

print(f"Root Mean Square Error (RMSE): {accuracy_scores['RMSE']}\n\n")

print(f"Weighted Absolute Percentage Error (WAPE): {accuracy_scores['WAPE']}\n\n")

print(f"Mean Absolute Percentage Error (MAPE): {accuracy_scores['MAPE']}\n\n")

print(f"Mean Absolute Scaled Error (MASE): {accuracy_scores['MASE']}\n")


# *********** Generate forecasts **********************
print('*********** Generate forecasts **********************')
FORECAST_NAME = "AUTOML4"

create_forecast_response = \
    forecast.create_forecast(ForecastName=FORECAST_NAME,
                             PredictorArn=predictor_arn)

forecast_arn = create_forecast_response['ForecastArn']
print(f"Waiting for Forecast with ARN {forecast_arn} to become ACTIVE. Depending on data size and predictor settings，it can take several hours to be ACTIVE.\n\nCurrent Status:")

status = util.wait(lambda: forecast.describe_forecast(ForecastArn=forecast_arn))

describe_forecast_response = forecast.describe_forecast(ForecastArn=forecast_arn)
print(f"\n\nThe Forecast with ARN {forecast_arn} is now {describe_forecast_response['Status']}.")
