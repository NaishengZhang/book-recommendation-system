# DSGA1004 - BIG DATA
## Final project
- Prof Brian McFee (bm106)
- Mayank Lamba (ml5711)
- Saumya Goyal (sg5290)

*Handout date*: 2019-04-24

*Submission deadline*: 2019-05-18


# Overview

In the final project, you will apply the tools you have learned in this class to build and evaluate a recommender system.  While the content of the final project involves recommender systems, it is intended more as an opportunity to integrate multiple techniques to solve a realistic, large-scale applied problem.

For this project, you are encouraged to work in groups of no more than 3 students.

Groups of 1--2 will need to implement one extension (described below) over the baseline project for full credit.

Groups of 3 will need to implement two extensions for full credit.

## The data set

On Dumbo's HDFS, you will find the following files in `hdfs:/user/bm106/pub/project`:

  - `cf_train.parquet`
  - `cf_validation.parquet`
  - `cf_test.parquet`
  
  - `metadata.parquet`
  - `features.parquet`
  - `tags.parquet`
  - `lyrics.parquet`
  
  
The first three files contain training, validation, and testing data for the collaborative filter.  Specifically, each file contains a table of triples `(user_id, track_id, count)` which measure implicit feedback derived from listening behavior.

The four additional files consist of supplementary data for each track (item) in the dataset.  You are not required to use any of these, but they may be helpful when implementing extensions to the baseline model.

## Project structure

TBA

## Extensions

For full credit, implement an extension on top of the baseline collaborative filter model.  The choice of extension is up to you, but here are some ideas:

  - *Fast search*: use a spatial data structure (e.g., LSH or partition trees) to implement accelerated search at query time.  For this, it is best to use an existing library such as `annoy` or `nmslib`, and you will need to export the model parameters from Spark to work in your chosen environment.  For full credit, you should provide a thorough evaluation of the efficiency gains provided by your spatial data structure over a brute-force search method.
  - *Alternative model formualtions*: the `AlternatingLeastSquares` model in Spark implements a particular form of implicit-feedback modeling, but you could change its behavior by modifying the entries in the utility matrix.  Conduct a thorough evaluation of different modification strategies (e.g., log compression, or dropping low count values) and their impact on overall accuracy.
  - *Cold-start*: using the supplementary data, build a model that can map observable feature data to the learned latent factor representation for items.  To evaluate its accuracy, simulate a cold-start scenario by holding out a subset of items during training (of the recommender model), and compare its performance to a full collaborative filter model.
  - *Error analysis*: after training the model, analyze the errors that it makes.  Are certain types of item over- or under-represented?  Make use of the supplementary metadata and tag information to inform your analysis.
  - *Exploration*: use the learned representation to develop a visualization of the items and users, e.g., using T-SNE or UMAP.

You are welcome to propose your own extension ideas, but they must be submitted in writing and approved by the course staff (Brian, Mayank, or Saumya) by 2019-05-06 at the latest.


# What to turn in

In addition to all of your code, produce a final report (not to exceed 4 pages), describing your baseline implementation, evaluation results, and extensions.  Your report should clearly identify the contributions of each member of your group.  If any additional software components were required in your project, your choices should be described and well motivated here.  

Include a PDF copy of your report in the github repository along with your code submission.

Any additional software components should be listed in `requirements.txt` (if using Python packages).


# Checklist

It will be helpful to commit your work in progress to the repository.  Toward this end, we recommend the following timeline:

- [ ] 2019/05/08: working baseline implementation 
- [ ] 2019/05/10: select extension(s)
- [ ] 2019/05/18: final submission
