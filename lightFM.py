#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    spark-submit lightFM.py hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv 0.01
    interactions = spark.read.csv("hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv")

    x = "goodreads_interactions"
    df.printSchema()
    user_id,book_id,is_read,rating,is_reviewed
    df = spark.read.csv(f'hdfs:/user/bm106/pub/goodreads/{x}.csv', header=True, schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')


    scp /Users/jonathan/Desktop/data/small/goodreads_interactions_poetry.json  nz862@dumbo.hpc.nyu.edu:/home/nz862/final
    scp /Users/jonathan/Desktop/data/als_train.py  nz862@dumbo.hpc.nyu.edu:/home/nz862/final
'''

import sys
import math
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
import numpy as np
import pyspark.sql.functions as F
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import preprocessing
from lightfm import LightFM
from scipy.sparse import csr_matrix 
from scipy.sparse import coo_matrix 
from sklearn.metrics import roc_auc_score
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import pickle
import re
import datetime
import time

def main(spark, data_file, percent_data):

    # Read data from parquet
    # interactions = spark.read.csv(data_file, header=True, inferSchema=True)
    # interactions.write.parquet('interactions.parquet')
    time_stamp = datetime.datetime.now()
    f = open("lightout.txt", "a")
    print("\nprogram start at:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), file=f) ##2017.02.19-14:03:20
    f.close()
    interactions_pq = spark.read.parquet('interactions.parquet')
    interactions_pq = interactions_pq.select('user_id', 'book_id', 'rating')
    interactions_pq = interactions_pq.filter(interactions_pq.rating > 0)
    interactions_pq.createOrReplaceTempView('interactions_pq')

    f = open("lightout.txt", "a")
    print("finish reading data", file=f)
    f.close()
    #filter: count of interactions > 10
    dsamp = spark.sql("SELECT user_id, count(user_id) as cnt from interactions_pq GROUP BY user_id having cnt > 10  order by cnt")
    dsamp.createOrReplaceTempView('dsamp')
    # data after previous filter;count > 10 in samples dataset
    samples = spark.sql("SELECT dsamp.user_id, interactions_pq.book_id, interactions_pq.rating from interactions_pq join dsamp on interactions_pq.user_id = dsamp.user_id")
    samples.createOrReplaceTempView('samples')

    # distinct user id
    user = samples.select('user_id').distinct()
    # downsample 20% of data
    user, drop = user.randomSplit([percent_data, 1-percent_data],2)

    #split data, tarin, validation,test
    a,b,c = user.randomSplit([0.6, 0.2, 0.2],2)
    a.createOrReplaceTempView('a')
    b.createOrReplaceTempView('b')
    c.createOrReplaceTempView('c')

    f = open("lightout.txt", "a")
    print("finish spliting", file=f)
    f.close()
    
    # raw tarin, validation,test
    train = spark.sql('''select samples.user_id, samples.book_id, samples.rating from samples join a on samples.user_id = a.user_id''')    
    validation = spark.sql('''select samples.user_id, samples.book_id, samples.rating from samples join b on samples.user_id = b.user_id''')
    test = spark.sql('''select samples.user_id, samples.book_id, samples.rating from samples join c on samples.user_id = c.user_id''')

    # For each validation user, use half of their interactions for training, and the other half should be held out for validation.
    window = Window.partitionBy(validation.user_id).orderBy(validation.book_id)
    val = (validation.select("user_id","book_id","rating",
                      F.row_number()
                      .over(window)
                      .alias("row_number")))
    val.createOrReplaceTempView('val')
    val_train = spark.sql('Select * from val where (row_number % 2) = 1')
    val_val = spark.sql('Select * from val where (row_number % 2) = 0')

    # same operations for test dataset
    window = Window.partitionBy(test.user_id).orderBy(test.book_id)
    test = (test.select("user_id","book_id","rating",
                      F.row_number()
                      .over(window)
                      .alias("row_number")))
    test.createOrReplaceTempView('test')
    test_train = spark.sql('Select * from test where (row_number % 2) = 1')
    test_test = spark.sql('Select * from test where (row_number % 2) = 0')

    # union:train + val_train + test_train
    # val_val
    # test_test
    val_train = val_train.select('user_id','book_id','rating')
    test_train = test_train.select('user_id','book_id','rating')
    test_test = test_test.select('user_id','book_id','rating')
    val_val = val_val.select('user_id','book_id','rating')
    train_total = train.union(val_train).union(test_train)
    train_total.createOrReplaceTempView('train_total')


    f = open("lightout.txt", "a")
    print("Starting convert to pandas dataframe", file=f)
    f.close()
    #convert spark dataframe to panda dataframe
    train = train_total.select("*").toPandas()
    test = test_test.select("*").toPandas()
    validation = val_val.select("*").toPandas()
    f = open("lightout.txt", "a")
    print("finish converting to pandas", file=f)
    f.close()


    f = open("lightout.txt", "a")
    print("Starting convert from dataframe to sparse matrix", file=f)
    f.close()
    #convert spark dataframe to sparse matrix
    train_matrix,validation_matrix,test_matrix = convert_to_matrix(train, validation, test)
    f = open("lightout.txt", "a")
    print("Finishing convert from dataframe to sparse matrix", file=f)
    f.close()
    #set the learning rate and learning schedule
    learning_rates = [0.01,0.05,0.1]
    learning_schedules = ['adagrad','adadelta']

    #Build and fit the model
    best_model,learning_rate,learning_shedule,max_score = tune_LightFM(train_matrix, validation_matrix,learning_rates,learning_schedules, 2)
    
    #Evaluate the model by using precision@k and ROC method
    f = open("lightout.txt", "a")
    print("k= 500, Train precision: %.2f" % precision_at_k(best_model, train_matrix, k=500).mean(),file=f)
    print("k= 500, Validation precision: %.2f" % precision_at_k(best_model, validation_matrix, k=500).mean(),file=f)
    print("k= 500, Test precision: %.2f" % precision_at_k(best_model, test_matrix, k=500).mean(),file=f)

    print("k= 100, Train precision: %.2f" % precision_at_k(best_model, train_matrix, k=100).mean(),file=f)
    print("k= 100, Validation precision: %.2f" % precision_at_k(best_model, validation_matrix, k=100).mean(),file=f)
    print("k= 100, Test precision: %.2f" % precision_at_k(best_model, test_matrix, k=100).mean(),file=f)

    print("k= 10, Train precision: %.2f" % precision_at_k(best_model, train_matrix, k=10).mean(),file=f)
    print("k= 10, Validation precision: %.2f" % precision_at_k(best_model, validation_matrix, k=10).mean(),file=f)
    print("k= 10, Test precision: %.2f" % precision_at_k(best_model, test_matrix, k=10).mean(),file=f)

    print("k= 5, Train precision: %.2f" % precision_at_k(best_model, train_matrix, k=5).mean(),file=f)
    print("k= 5, Validation precision: %.2f" % precision_at_k(best_model, validation_matrix, k=5).mean(),file=f)
    print("k= 5, Test precision: %.2f" % precision_at_k(best_model, test_matrix, k=5).mean(),file=f)
    score = auc_score(best_model, test_matrix,num_threads=1).mean()
    print('The auc_score is {} when learning rate = {} and learning schedule is {}'.format(score, learning_rate, learning_shedule),file=f)

    time_stamp = datetime.datetime.now()
    print("program end at:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), file=f) 
    f.close()

def tune_LightFM(train_data, validation_data, learning_rates, learning_schedules,num_threads):
    """
    learning_rates = [0.01,0.05,0.1]
    learning_schedules = ['adagrad','adadelta']
    evaluation method: ROC
    """
    # initial
    time_stamp = datetime.datetime.now()
    f = open("lightout.txt", "a")
    print("fine tuning program start at:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'),file=f) 
    f.close()
    start_time = time.time()
    max_score = 0
    best_lr = -1
    best_ls = ""
    best_model = None
    f = open("lightout.txt", "a")
    for learning_rate in learning_rates:
        for learning_schedule in learning_schedules:
            # get LightFM model
            time_stamp = datetime.datetime.now()
            print("program start fitting at:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'),file=f)
            model = LightFM(learning_rate=learning_rate,learning_schedule=learning_schedule, loss='bpr')
            # train LightFM model
            model.fit(train_data, epochs=10)
            print('finising the fit which the parameters is learning rate = {} and learning schedule is {}'.format(learning_rate, learning_schedule),file=f)
            # evaluate the model by computing the RMSE on the validation data
            score = auc_score(model, validation_data, num_threads=num_threads).mean()
            print('The score is {} when learning rate = {} and learning schedule is {}'.format(score, learning_rate, learning_schedule),file=f)
            if score > max_score:
                max_score = score
                best_lr = learning_rate
                best_ls = learning_schedule
                best_model = model
    print('\nWe can get the best model when the learning rate ={} and '
          'the learning schedule is {}, the best score = {}'.format(learning_rate, learning_schedule,max_score),file=f)
    print("--- Run time:  {} mins ---".format((time.time() - start_time)/60),file=f)
    f.close()
    return best_model,learning_rate,learning_schedule,max_score

#This function is to transfer the dataframe to sparse matrix
def convert_to_matrix(train, validation, test):
    id_cols = ['user_id', 'book_id']
    trans_cat_train = dict()
    trans_cat_test = dict()
    trans_cat_val = dict()
    test_df = test[(test['user_id'].isin(train['user_id'])) & (test['book_id'].isin(train['book_id']))]
    val_df = validation[(validation['user_id'].isin(train['user_id'])) & (validation['book_id'].isin(train['book_id']))]
    for k in id_cols:
        cate_enc = preprocessing.LabelEncoder() #normalize the rating label
        trans_cat_train[k] = cate_enc.fit_transform(train[k].values)
        trans_cat_test[k] = cate_enc.transform(test_df[k].values)
        trans_cat_val[k] = cate_enc.transform(val_df[k].values)
        
    cate_enc = preprocessing.LabelEncoder()
    ratings = dict()
    ratings['train'] = cate_enc.fit_transform(train.rating)
    ratings['val'] = cate_enc.transform(val_df.rating)
    ratings['test'] = cate_enc.transform(test_df.rating)
    n_users = len(np.unique(trans_cat_train['user_id']))
    n_items = len(np.unique(trans_cat_train['book_id']))
    
    train_matrix = coo_matrix((ratings['train'], (trans_cat_train['user_id'], \
                                                          trans_cat_train['book_id'])) \
                                      , shape=(n_users, n_items))
    test_matrix = coo_matrix((ratings['test'], (trans_cat_test['user_id'], \
                                                        trans_cat_test['book_id'])) \
                                     , shape=(n_users, n_items))
    validation_matrix = coo_matrix((ratings['val'], (trans_cat_val['user_id'], \
                                                        trans_cat_val['book_id'])) \
                                     , shape=(n_users, n_items))
    return train_matrix, validation_matrix, test_matrix

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('als_train').getOrCreate()
    memory = "15g" #for local but for cluster did a fixed number
    spark = (SparkSession.builder
             .appName('lightFM')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # Get the filename from the command line
    data_file = sys.argv[1]
    # data percent
    percent_data = float(sys.argv[2])
    # And the location to store the trained model
    # model_file = sys.argv[2]

    # Call our main routine
    # main(spark, data_file, model_file)
    main(spark, data_file, percent_data)
