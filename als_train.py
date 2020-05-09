#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    spark-submit als_train.py data/small/goodreads_interactions_poetry.json
    spark-submit als_train.py hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv 0.01
    interactions = spark.read.csv("hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv")

    x = "goodreads_interactions"
    df = spark.read.csv(f'hdfs:/user/bm106/pub/goodreads/{x}.csv', header=True, inferSchema=True)
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
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer,IndexToString 
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
import datetime

def main(spark, data_file, percent_data):

    # Read data from parquet
    # interactions = spark.read.csv(data_file, header=True, inferSchema=True)
    # interactions.write.parquet('interactions.parquet')
    time_stamp = datetime.datetime.now()
    f = open("out.txt", "a")
    print("program start at:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), file=f) ##2017.02.19-14:03:20
    f.close()
    interactions_pq = spark.read.parquet('interactions.parquet')
    interactions_pq = interactions_pq.select('user_id', 'book_id', 'rating')
    interactions_pq = interactions_pq.filter(interactions_pq.rating > 0)
    interactions_pq.createOrReplaceTempView('interactions_pq')

    f = open("out.txt", "a")
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

    f = open("out.txt", "a")
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
    train_total = train.union(val_train).union(test_train)
    train_total.createOrReplaceTempView('train_total')
    
    # convert string to numeric
    # train_new = convert(train_total)
    # val_val_new = convert(val_val)
    # test_test_new = convert(test_test)

    f = open("out.txt", "a")
    print("finish convert", file=f)
    f.close()

    # tuning parameter and train model
    ranks = [5,10,15,20,25,30]
    regParams = [0.01, 0.05, 0.1, 1]
    f = open("out.txt", "a")
    print("start fit model", file=f)
    f.close()
    ALSmodel,rank,regParams,rmse = tune_ALS(train_total, val_val, 5, regParams, ranks)
    print('final rank = {}, regParams = {}, rmse = {}'.format(rank,regParams,rmse))

    # predict test dataset
    predictions = ALSmodel.transform(test_test)

    # user_id_index,book_id_index,rating,recommendations
    # test_test_new: book rating
    windowSpec = Window.partitionBy('user_id').orderBy(col('rating').desc())
    perUserActualItemsDF = (test_test
               .select('user_id', 'book_id', 'rating', F.rank().over(windowSpec).alias('rank'))
               .where(f'rank <= {500} and rating > {0}') # rank = 500
               .groupBy('user_id')
               .agg(expr('collect_list(book_id) as recommendations')))

    # prediction: recommend book rating
    windowSpec = Window.partitionBy('user_id').orderBy(col('prediction').desc())
    perUserPredictedItemsDF = (predictions
               .select('user_id', 'book_id', 'prediction', F.rank().over(windowSpec).alias('rank'))
               .where(f'rank <= {500} and rating > {0}')
               .groupBy('user_id')
               .agg(expr('collect_list(book_id) as recommendations')))


    # select recommendations,recommendations
    perUserItemsRDD = perUserPredictedItemsDF.join(perUserActualItemsDF, 'user_id') \
        .rdd \
        .map(lambda row: (row[1], row[2]))

    rankingMetrics = RankingMetrics(perUserItemsRDD)

    time_stamp = datetime.datetime.now()
    f = open("out.txt", "a")
    print('\nMAP = {}, precisionAtk = {}, ndcgAt = {}'\
        .format(rankingMetrics.meanAveragePrecision, rankingMetrics.precisionAt(500),rankingMetrics.ndcgAt(500)), file=f)
    print("program end at:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), file=f) ##2017.02.19-14:03:20
    f.close()


def convert(dataframe):
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(dataframe) for column in list(set(dataframe.columns))]
    pipeline = Pipeline(stages=indexers)
    dataframe_new = pipeline.fit(dataframe).transform(dataframe)
    return dataframe_new

def tune_ALS(train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    train_data: spark DF with columns ['user_Id', 'book_id', 'rating']
    
    validation_data: spark DF with columns ['user_Id', 'book_id', 'rating']
    
    maxIter: int, max number of learning iterations
    
    regParams: list of float, one dimension of hyper-param tuning grid
    
    ranks: list of float, one dimension of hyper-param tuning grid
    
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = ALS().setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            als.setUserCol("user_id").setItemCol("book_id").setRatingCol("rating").setColdStartStrategy("drop")
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            f = open("out.txt", "a")
            # print(predictions.show(50), file=f)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse), file=f)
            f.close()
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    f = open("out.txt", "a")
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization), file=f)
    f.close()   
    return best_model,best_rank,best_regularization,min_error


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('als_train').getOrCreate()
    memory = "15g" #for local but for cluster did a fixed number
    spark = (SparkSession.builder
             .appName('als_train1')
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
