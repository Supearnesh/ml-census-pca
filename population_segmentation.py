#!/usr/bin/env python
# coding: utf-8

# # Population Segmentation with SageMaker
# 
# In this notebook, you'll employ two, unsupervised learning algorithms to do **population segmentation**. Population segmentation aims to find natural groupings in population data that reveal some feature-level similarities between different regions in the US.
# 
# Using **principal component analysis** (PCA) you will reduce the dimensionality of the original census data. Then, you'll use **k-means clustering** to assign each US county to a particular cluster based on where a county lies in component space. How each cluster is arranged in component space can tell you which US counties are most similar and what demographic traits define that similarity; this information is most often used to inform targeted, marketing campaigns that want to appeal to a specific group of people. This cluster information is also useful for learning more about a population by revealing patterns between regions that you otherwise may not have noticed.
# 
# ### US Census Data
# 
# You'll be using data collected by the [US Census](https://en.wikipedia.org/wiki/United_States_Census), which aims to count the US population, recording demographic traits about labor, age, population, and so on, for each county in the US. The bulk of this notebook was taken from an existing SageMaker example notebook and [blog post](https://aws.amazon.com/blogs/machine-learning/analyze-us-census-data-for-population-segmentation-using-amazon-sagemaker/), and I've broken it down further into demonstrations and exercises for you to complete.
# 
# ### Machine Learning Workflow
# 
# To implement population segmentation, you'll go through a number of steps:
# * Data loading and exploration
# * Data cleaning and pre-processing 
# * Dimensionality reduction with PCA
# * Feature engineering and data transformation
# * Clustering transformed data with k-means
# * Extracting trained model attributes and visualizing k clusters
# 
# These tasks make up a complete, machine learning workflow from data loading and cleaning to model deployment. Each exercise is designed to give you practice with part of the machine learning workflow, and to demonstrate how to use SageMaker tools, such as built-in data management with S3 and built-in algorithms.
# 
# ---

# First, import the relevant libraries into this SageMaker notebook. 

# In[1]:


# data managing and display libs
import pandas as pd
import numpy as np
import os
import io

import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# sagemaker libraries
import boto3
import sagemaker


# ## Loading the Data from Amazon S3
# 
# This particular dataset is already in an Amazon S3 bucket; you can load the data by pointing to this bucket and getting a data file by name. 
# 
# > You can interact with S3 using a `boto3` client.

# In[3]:


# boto3 client to get S3 data
s3_client = boto3.client('s3')
bucket_name='aws-ml-blog-sagemaker-census-segmentation'


# Take a look at the contents of this bucket; get a list of objects that are contained within the bucket and print out the names of the objects. You should see that there is one file, 'Census_Data_for_SageMaker.csv'.

# In[4]:


# get a list of objects in the bucket
obj_list=s3_client.list_objects(Bucket=bucket_name)

# print object(s)in S3 bucket
files=[]
for contents in obj_list['Contents']:
    files.append(contents['Key'])
    
print(files)


# In[5]:


# there is one file --> one key
file_name=files[0]

print(file_name)


# Retrieve the data file from the bucket with a call to `client.get_object()`.

# In[6]:


# get an S3 object by passing in the bucket and file name
data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)

# what info does the object contain?
display(data_object)


# In[7]:


# information is in the "Body" of the object
data_body = data_object["Body"].read()
print('Data type: ', type(data_body))


# This is a `bytes` datatype, which you can read it in using [io.BytesIO(file)](https://docs.python.org/3/library/io.html#binary-i-o).

# In[8]:


# read in bytes data
data_stream = io.BytesIO(data_body)

# create a dataframe
counties_df = pd.read_csv(data_stream, header=0, delimiter=",") 
counties_df.head()


# ## Exploratory Data Analysis (EDA)
# 
# Now that you've loaded in the data, it is time to clean it up, explore it, and pre-process it. Data exploration is one of the most important parts of the machine learning workflow because it allows you to notice any initial patterns in data distribution and features that may inform how you proceed with modeling and clustering the data.
# 
# ### EXERCISE: Explore data & drop any incomplete rows of data
# 
# When you first explore the data, it is good to know what you are working with. How many data points and features are you starting with, and what kind of information can you get at a first glance? In this notebook, you're required to use complete data points to train a model. So, your first exercise will be to investigate the shape of this data and implement a simple, data cleaning step: dropping any incomplete rows of data.
# 
# You should be able to answer the **question**: How many data points and features are in the original, provided dataset? (And how many points are left after dropping any incomplete rows?)

# In[9]:


# print out stats about data
# rows = data, cols = features
print('(orig) rows, cols: ', counties_df.shape)

# drop any incomplete rows of data, and create a new df
clean_counties_df = counties_df.dropna(axis=0)
print('(clean) rows, cols: ', clean_counties_df.shape)


# ### EXERCISE: Create a new DataFrame, indexed by 'State-County'
# 
# Eventually, you'll want to feed these features into a machine learning model. Machine learning models need numerical data to learn from and not categorical data like strings (State, County). So, you'll reformat this data such that it is indexed by region and you'll also drop any features that are not useful for clustering.
# 
# To complete this task, perform the following steps, using your *clean* DataFrame, generated above:
# 1. Combine the descriptive columns, 'State' and 'County', into one, new categorical column, 'State-County'. 
# 2. Index the data by this unique State-County name.
# 3. After doing this, drop the old State and County columns and the CensusId column, which does not give us any meaningful demographic information.
# 
# After completing this task, you should have a DataFrame with 'State-County' as the index, and 34 columns of numerical data for each county. You should get a resultant DataFrame that looks like the following (truncated for display purposes):
# ```
#                 TotalPop	 Men	  Women	Hispanic	...
#                 
# Alabama-Autauga	55221	 26745	28476	2.6         ...
# Alabama-Baldwin	195121	95314	99807	4.5         ...
# Alabama-Barbour	26932	 14497	12435	4.6         ...
# ...
# 
# ```

# In[10]:


# index data by 'State-County'
clean_counties_df.index=clean_counties_df['State'] + "-" + clean_counties_df['County']
clean_counties_df.head()


# In[11]:


# drop the old State and County columns, and the CensusId column
# clean df should be modified or created anew
drop = ['CensusId', 'State', 'County']
clean_counties_df = clean_counties_df.drop(columns=drop)
clean_counties_df.head()


# Now, what features do you have to work with?

# In[12]:


# features
features_list = clean_counties_df.columns.values
print('Features: \n', features_list)


# ## Visualizing the Data
# 
# In general, you can see that features come in a variety of ranges, mostly percentages from 0-100, and counts that are integer values in a large range. Let's visualize the data in some of our feature columns and see what the distribution, over all counties, looks like.
# 
# The below cell displays **histograms**, which show the distribution of data points over discrete feature ranges. The x-axis represents the different bins; each bin is defined by a specific range of values that a feature can take, say between the values 0-5 and 5-10, and so on. The y-axis is the frequency of occurrence or the number of county data points that fall into each bin. I find it helpful to use the y-axis values for relative comparisons between different features.
# 
# Below, I'm plotting a histogram comparing methods of commuting to work over all of the counties. I just copied these feature names from the list of column names, printed above. I also know that all of these features are represented as percentages (%) in the original data, so the x-axes of these plots will be comparable.

# In[13]:


# transportation (to work)
transport_list = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']
n_bins = 75 # can decrease to get a wider bin (or vice versa)

for column_name in transport_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name], bins=n_bins)
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()


# ### EXERCISE: Create histograms of your own
# 
# Commute transportation method is just one category of features. If you take a look at the 34 features, you can see data on profession, race, income, and more. Display a set of histograms that interest you!
# 

# In[14]:


# employment metrics
my_list = ['PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']
n_bins = 50 # define n_bins

# histogram creation code is similar to above
for column_name in my_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name], bins=n_bins)
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()


# ### EXERCISE: Normalize the data
# 
# You need to standardize the scale of the numerical columns in order to consistently compare the values of different features. You can use a [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) to transform the numerical values so that they all fall between 0 and 1.

# In[15]:


# scale numerical features into a normalized range, 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# store them in this dataframe
counties_scaled = pd.DataFrame(scaler.fit_transform(clean_counties_df.astype(float)))

# get same features and State-County indices
counties_scaled.columns=clean_counties_df.columns
counties_scaled.index=clean_counties_df.index

counties_scaled.head()


# In[16]:


counties_scaled.describe()


# ---
# # Data Modeling
# 
# 
# Now, the data is ready to be fed into a machine learning model!
# 
# Each data point has 34 features, which means the data is 34-dimensional. Clustering algorithms rely on finding clusters in n-dimensional feature space. For higher dimensions, an algorithm like k-means has a difficult time figuring out which features are most important, and the result is, often, noisier clusters.
# 
# Some dimensions are not as important as others. For example, if every county in our dataset has the same rate of unemployment, then that particular feature doesn’t give us any distinguishing information; it will not help t separate counties into different groups because its value doesn’t *vary* between counties.
# 
# > Instead, we really want to find the features that help to separate and group data. We want to find features that cause the **most variance** in the dataset!
# 
# So, before I cluster this data, I’ll want to take a dimensionality reduction step. My aim will be to form a smaller set of features that will better help to separate our data. The technique I’ll use is called PCA or **principal component analysis**
# 
# ## Dimensionality Reduction
# 
# PCA attempts to reduce the number of features within a dataset while retaining the “principal components”, which are defined as *weighted*, linear combinations of existing features that are designed to be linearly independent and account for the largest possible variability in the data! You can think of this method as taking many features and combining similar or redundant features together to form a new, smaller feature set.
# 
# We can reduce dimensionality with the built-in SageMaker model for PCA.

# ### Roles and Buckets
# 
# > To create a model, you'll first need to specify an IAM role, and to save the model attributes, you'll need to store them in an S3 bucket.
# 
# The `get_execution_role` function retrieves the IAM role you created at the time you created your notebook instance. Roles are essentially used to manage permissions and you can read more about that [in this documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). For now, know that we have a FullAccess notebook, which allowed us to access and download the census data stored in S3.
# 
# You must specify a bucket name for an S3 bucket in your account where you want SageMaker model parameters to be stored. Note that the bucket must be in the same region as this notebook. You can get a default S3 bucket, which automatically creates a bucket for you and in your region, by storing the current SageMaker session and calling `session.default_bucket()`.

# In[17]:


from sagemaker import get_execution_role

session = sagemaker.Session() # store the current SageMaker session

# get IAM role
role = get_execution_role()
print(role)


# In[18]:


# get default bucket
bucket_name = session.default_bucket()
print(bucket_name)
print()


# ## Define a PCA Model
# 
# To create a PCA model, I'll use the built-in SageMaker resource. A SageMaker estimator requires a number of parameters to be specified; these define the type of training instance to use and the model hyperparameters. A PCA model requires the following constructor arguments:
# 
# * role: The IAM role, which was specified, above.
# * train_instance_count: The number of training instances (typically, 1).
# * train_instance_type: The type of SageMaker instance for training.
# * num_components: An integer that defines the number of PCA components to produce.
# * sagemaker_session: The session used to train on SageMaker.
# 
# Documentation on the PCA model can be found [here](http://sagemaker.readthedocs.io/en/latest/pca.html).
# 
# Below, I first specify where to save the model training data, the `output_path`.

# In[19]:


# define location to store model artifacts
prefix = 'counties'

output_path='s3://{}/{}/'.format(bucket_name, prefix)

print('Training artifacts will be uploaded to: {}'.format(output_path))


# In[20]:


# define a PCA model
from sagemaker import PCA

# this is current features - 1
# you'll select only a portion of these to use, later
N_COMPONENTS=33

pca_SM = PCA(role=role,
             train_instance_count=1,
             train_instance_type='ml.c4.xlarge',
             output_path=output_path, # specified, above
             num_components=N_COMPONENTS, 
             sagemaker_session=session)


# ### Convert data into a RecordSet format
# 
# Next, prepare the data for a built-in model by converting the DataFrame to a numpy array of float values.
# 
# The *record_set* function in the SageMaker PCA model converts a numpy array into a **RecordSet** format that is the required format for the training input data. This is a requirement for _all_ of SageMaker's built-in models. The use of this data type is one of the reasons that allows training of models within Amazon SageMaker to perform faster, especially for large datasets.

# In[21]:


# convert df to np array
train_data_np = counties_scaled.values.astype('float32')

# convert to RecordSet format
formatted_train_data = pca_SM.record_set(train_data_np)


# ## Train the model
# 
# Call the fit function on the PCA model, passing in our formatted, training data. This spins up a training instance to perform the training job.
# 
# Note that it takes the longest to launch the specified training instance; the fitting itself doesn't take much time.

# In[22]:


get_ipython().run_cell_magic('time', '', '\n# train the PCA mode on the formatted data\npca_SM.fit(formatted_train_data)')


# ## Accessing the PCA Model Attributes
# 
# After the model is trained, we can access the underlying model parameters.
# 
# ### Unzip the Model Details
# 
# Now that the training job is complete, you can find the job under **Jobs** in the **Training**  subsection  in the Amazon SageMaker console. You can find the job name listed in the training jobs. Use that job name in the following code to specify which model to examine.
# 
# Model artifacts are stored in S3 as a TAR file; a compressed file in the output path we specified + 'output/model.tar.gz'. The artifacts stored here can be used to deploy a trained model.

# In[23]:


# Get the name of the training job, it's suggested that you copy-paste
# from the notebook or from a specific job in the AWS console

training_job_name='pca-2020-04-30-00-02-39-867'

# where the model is saved, by default
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')
print(model_key)

# download and unzip model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzipping as model_algo-1
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# ### MXNet Array
# 
# Many of the Amazon SageMaker algorithms use MXNet for computational speed, including PCA, and so the model artifacts are stored as an array. After the model is unzipped and decompressed, we can load the array using MXNet.
# 
# You can take a look at the MXNet [documentation, here](https://aws.amazon.com/mxnet/).

# In[24]:


import mxnet as mx

# loading the unzipped artifacts
pca_model_params = mx.ndarray.load('model_algo-1')

# what are the params
print(pca_model_params)


# ## PCA Model Attributes
# 
# Three types of model attributes are contained within the PCA model.
# 
# * **mean**: The mean that was subtracted from a component in order to center it.
# * **v**: The makeup of the principal components; (same as ‘components_’ in an sklearn PCA model).
# * **s**: The singular values of the components for the PCA transformation. This does not exactly give the % variance from the original feature space, but can give the % variance from the projected feature space.
#     
# We are only interested in v and s. 
# 
# From s, we can get an approximation of the data variance that is covered in the first `n` principal components. The approximate explained variance is given by the formula: the sum of squared s values for all top n components over the sum over squared s values for _all_ components:
# 
# \begin{equation*}
# \frac{\sum_{n}^{ } s_n^2}{\sum s^2}
# \end{equation*}
# 
# From v, we can learn more about the combinations of original features that make up each principal component.
# 

# In[25]:


# get selected params
s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())


# ## Data Variance
# 
# Our current PCA model creates 33 principal components, but when we create new dimensionality-reduced training data, we'll only select a few, top n components to use. To decide how many top components to include, it's helpful to look at how much **data variance** the components capture. For our original, high-dimensional data, 34 features captured 100% of our data variance. If we discard some of these higher dimensions, we will lower the amount of variance we can capture.
# 
# ### Tradeoff: dimensionality vs. data variance
# 
# As an illustrative example, say we have original data in three dimensions. So, three dimensions capture 100% of our data variance; these dimensions cover the entire spread of our data. The below images are taken from the PhD thesis,  [“Approaches to analyse and interpret biological profile data”](https://publishup.uni-potsdam.de/opus4-ubp/frontdoor/index/index/docId/696) by Matthias Scholz, (2006, University of Potsdam, Germany).
# 
# <img src='notebook_ims/3d_original_data.png' width=35% />
# 
# Now, you may also note that most of this data seems related; it falls close to a 2D plane, and just by looking at the spread of the data, we  can visualize that the original, three dimensions have some correlation. So, we can instead choose to create two new dimensions, made up of linear combinations of the original, three dimensions. These dimensions are represented by the two axes/lines, centered in the data. 
# 
# <img src='notebook_ims/pca_2d_dim_reduction.png' width=70% />
# 
# If we project this in a new, 2D space, we can see that we still capture most of the original data variance using *just* two dimensions. There is a tradeoff between the amount of variance we can capture and the number of component-dimensions we use to represent our data.
# 
# When we select the top n components to use in a new data model, we'll typically want to include enough components to capture about 80-90% of the original data variance. In this project, we are looking at generalizing over a lot of data and we'll aim for about 80% coverage.

# **Note**: The _top_ principal components, with the largest s values, are actually at the end of the s DataFrame. Let's print out the s values for the top n, principal components.

# In[26]:


# looking at top 5 components
n_principal_components = 5

start_idx = N_COMPONENTS - n_principal_components  # 33-n

# print a selection of s
print(s.iloc[start_idx:, :])


# ### EXERCISE: Calculate the explained variance
# 
# In creating new training data, you'll want to choose the top n principal components that account for at least 80% data variance. 
# 
# Complete a function, `explained_variance` that takes in the entire array `s` and a number of top principal components to consider. Then return the approximate, explained variance for those top n components. 
# 
# For example, to calculate the explained variance for the top 5 components, calculate s squared for *each* of the top 5 components, add those up and normalize by the sum of *all* squared s values, according to this formula:
# 
# \begin{equation*}
# \frac{\sum_{5}^{ } s_n^2}{\sum s^2}
# \end{equation*}
# 
# > Using this function, you should be able to answer the **question**: What is the smallest number of principal components that captures at least 80% of the total variance in the dataset?

# In[27]:


# Calculate the explained variance for the top n principal components
# you may assume you have access to the global var N_COMPONENTS
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    start_idx = N_COMPONENTS - n_top_components  # 33-n
    
    # calculate approximate variance
    exp_variance = np.square(s.iloc[start_idx:,:]).sum() / np.square(s).sum()
    return exp_variance[0]


# ### Test Cell
# 
# Test out your own code by seeing how it responds to different inputs; does it return a reasonable value for the single, top component? What about for the top 5 components?

# In[28]:


# test cell
n_top_components = 7 # select a value for the number of top components

# calculate the explained variance
exp_variance = explained_variance(s, n_top_components)
print('Explained variance: ', exp_variance)


# As an example, you should see that the top principal component accounts for about 32% of our data variance! Next, you may be wondering what makes up this (and other components); what linear combination of features make these components so influential in describing the spread of our data?
# 
# Below, let's take a look at our original features and use that as a reference.

# In[29]:


# features
features_list = counties_scaled.columns.values
print('Features: \n', features_list)


# ## Component Makeup
# 
# We can now examine the makeup of each PCA component based on **the weightings of the original features that are included in the component**. The following code shows the feature-level makeup of the first component.
# 
# Note that the components are again ordered from smallest to largest and so I am getting the correct rows by calling N_COMPONENTS-1 to get the top, 1, component.

# In[30]:


import seaborn as sns

def display_component(v, features_list, component_num, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()


# In[31]:


# display makeup of first component
num=2
display_component(v, counties_scaled.columns.values, component_num=num, n_weights=10)


# # Deploying the PCA Model
# 
# We can now deploy this model and use it to make "predictions". Instead of seeing what happens with some test data, we'll actually want to pass our training data into the deployed endpoint to create principal components for each data point. 
# 
# Run the cell below to deploy/host this model on an instance_type that we specify.

# In[32]:


get_ipython().run_cell_magic('time', '', "# this takes a little while, around 7mins\npca_predictor = pca_SM.deploy(initial_instance_count=1, \n                              instance_type='ml.t2.medium')")


# We can pass the original, numpy dataset to the model and transform the data using the model we created. Then we can take the largest n components to reduce the dimensionality of our data.

# In[33]:


# pass np train data to the PCA model
train_pca = pca_predictor.predict(train_data_np)


# In[34]:


# check out the first item in the produced training features
data_idx = 0
print(train_pca[data_idx])


# ### EXERCISE: Create a transformed DataFrame
# 
# For each of our data points, get the top n component values from the list of component data points, returned by our predictor above, and put those into a new DataFrame.
# 
# You should end up with a DataFrame that looks something like the following:
# ```
#                      c_1	     c_2	       c_3	       c_4	      c_5	   ...
# Alabama-Autauga	-0.060274	0.160527	-0.088356	 0.120480	-0.010824	...
# Alabama-Baldwin	-0.149684	0.185969	-0.145743	-0.023092	-0.068677	...
# Alabama-Barbour	0.506202	 0.296662	 0.146258	 0.297829	0.093111	...
# ...
# ```

# In[35]:


# create dimensionality-reduced data
def create_transformed_df(train_pca, counties_scaled, n_top_components):
    ''' Return a dataframe of data points with component features. 
        The dataframe should be indexed by State-County and contain component values.
        :param train_pca: A list of pca training data, returned by a PCA model.
        :param counties_scaled: A dataframe of normalized, original features.
        :param n_top_components: An integer, the number of top components to use.
        :return: A dataframe, indexed by State-County, with n_top_component values as columns.        
     '''
    # create a dataframe of component features, indexed by State-County
    # with columns = component values
    counties_transformed = pd.DataFrame()
    
    # for each of the new, transformed data points
    # append the component values to the dataframe
    for data in train_pca:
        # get component values for each data point
        components = data.label['projection'].float32_tensor.values
        counties_transformed = counties_transformed.append([list(components)])
    
    # index by county, just like counties_scaled
    counties_transformed.index = counties_scaled.index
    
    # keep only the top n components
    start_idx = N_COMPONENTS - n_top_components
    counties_transformed = counties_transformed.iloc[:,start_idx:]
    
    # reverse columns, component order
    return counties_transformed.iloc[:,::-1]


# Now we can create a dataset where each county is described by the top n principle components that we analyzed earlier. Each of these components is a linear combination of the original feature space. We can interpret each of these components by analyzing the makeup of the component, shown previously.
# 
# ### Define the `top_n` components to use in this transformed data
# 
# Your code should return data, indexed by 'State-County' and with as many columns as `top_n` components.
# 
# You can also choose to add descriptive column names for this data; names that correspond to the component number or feature-level makeup.

# In[36]:


## Specify top n
top_n = 7

# call your function and create a new dataframe
counties_transformed = create_transformed_df(train_pca, counties_scaled, n_top_components=top_n)

# columns for PCA
PCA_list = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7']
counties_transformed.columns = PCA_list

# print result
counties_transformed.head()


# ### Delete the Endpoint!
# 
# Now that we've deployed the mode and created our new, transformed training data, we no longer need the PCA endpoint.
# 
# As a clean up step, you should always delete your endpoints after you are done using them (and if you do not plan to deploy them to a website, for example).

# In[37]:


# delete predictor endpoint
session.delete_endpoint(pca_predictor.endpoint)


# ---
# # Population Segmentation 
# 
# Now, you’ll use the unsupervised clustering algorithm, k-means, to segment counties using their PCA attributes, which are in the transformed DataFrame we just created. K-means is a clustering algorithm that identifies clusters of similar data points based on their component makeup. Since we have ~3000 counties and 34 attributes in the original dataset, the large feature space may have made it difficult to cluster the counties effectively. Instead, we have reduced the feature space to 7 PCA components, and we’ll cluster on this transformed dataset.

# ### EXERCISE: Define a k-means model
# 
# Your task will be to instantiate a k-means model. A `KMeans` estimator requires a number of parameters to be instantiated, which allow us to specify the type of training instance to use, and the model hyperparameters. 
# 
# You can read about the required parameters, in the [`KMeans` documentation](https://sagemaker.readthedocs.io/en/stable/kmeans.html); note that not all of the possible parameters are required.
# 

# ### Choosing a "Good" K
# 
# One method for choosing a "good" k, is to choose based on empirical data. A bad k would be one so *high* that only one or two very close data points are near it, and another bad k would be one so *low* that data points are really far away from the centers.
# 
# You want to select a k such that data points in a single cluster are close together but that there are enough clusters to effectively separate the data. You can approximate this separation by measuring how close your data points are to each cluster center; the average centroid distance between cluster points and a centroid. After trying several values for k, the centroid distance typically reaches some "elbow"; it stops decreasing at a sharp rate and this indicates a good value of k. The graph below indicates the average centroid distance for value of k between 5 and 12.
# 
# <img src='notebook_ims/elbow_graph.png' width=50% />
# 
# A distance elbow can be seen around 8 when the distance starts to increase and then decrease at a slower rate. This indicates that there is enough separation to distinguish the data points in each cluster, but also that you included enough clusters so that the data points aren’t *extremely* far away from each cluster.

# In[38]:


# define a KMeans estimator
from sagemaker import KMeans

NUM_CLUSTERS = 8

kmeans = KMeans(role=role,
                train_instance_count=1,
                train_instance_type='ml.c4.xlarge',
                output_path=output_path, # using the same output path defined earlier
                k=NUM_CLUSTERS)


# ### EXERCISE: Create formatted, k-means training data
# 
# Just as before, you should convert the `counties_transformed` df into a numpy array and then into a RecordSet. This is the required format for passing training data into a `KMeans` model.

# In[39]:


# convert the transformed dataframe into record_set data
kmeans_train_data_np = counties_transformed.values.astype('float32')
kmeans_formatted_data = kmeans.record_set(kmeans_train_data_np)


# ### EXERCISE: Train the k-means model
# 
# Pass in the formatted training data and train the k-means model.

# In[40]:


get_ipython().run_cell_magic('time', '', '# train kmeans\nkmeans.fit(kmeans_formatted_data)')


# ### EXERCISE: Deploy the k-means model
# 
# Deploy the trained model to create a `kmeans_predictor`.
# 

# In[41]:


get_ipython().run_cell_magic('time', '', "# deploy the model to create a predictor\nkmeans_predictor = kmeans.deploy(initial_instance_count=1,\n                                 instance_type='ml.t2.medium')")


# ### EXERCISE: Pass in the training data and assign predicted cluster labels
# 
# After deploying the model, you can pass in the k-means training data, as a numpy array, and get resultant, predicted cluster labels for each data point.

# In[42]:


# get the predicted clusters for all the kmeans training data
cluster_info = kmeans_predictor.predict(kmeans_train_data_np)


# ## Exploring the resultant clusters
# 
# The resulting predictions should give you information about the cluster that each data point belongs to.
# 
# You should be able to answer the **question**: which cluster does a given data point belong to?

# In[43]:


# print cluster info for first data point
data_idx = 0

print('County is: ', counties_transformed.index[data_idx])
print()
print(cluster_info[data_idx])


# ### Visualize the distribution of data over clusters
# 
# Get the cluster labels for each of our data points (counties) and visualize the distribution of points over each cluster.

# In[44]:


# get all cluster labels
cluster_labels = [c.label['closest_cluster'].float32_tensor.values[0] for c in cluster_info]


# In[45]:


# count up the points in each cluster
cluster_df = pd.DataFrame(cluster_labels)[0].value_counts()

print(cluster_df)


# In[48]:


# another method of visualizing the distribution
# display a histogram of cluster counts
ax = plt.subplots(figsize=(6,3))
ax = plt.hist(cluster_labels, bins=8, range=(-0.5, 7.5), color='blue', rwidth=0.5)

title = 'Histogram of Cluster Counts'
plt.title(title, fontsize=12)
plt.show()


# Now, you may be wondering, what do each of these clusters tell us about these data points? To improve explainability, we need to access the underlying model to get the cluster centers. These centers will help describe which features characterize each cluster.

# ### Delete the Endpoint!
# 
# Now that you've deployed the k-means model and extracted the cluster labels for each data point, you no longer need the k-means endpoint.

# In[49]:


# delete kmeans endpoint
session.delete_endpoint(kmeans_predictor.endpoint)


# ---
# # Model Attributes & Explainability
# 
# Explaining the result of the modeling is an important step in making use of our analysis. By combining PCA and k-means, and the information contained in the model attributes within a SageMaker trained model, you can learn about a population and remark on some patterns you've found, based on the data.

# ### EXERCISE: Access the k-means model attributes
# 
# Extract the k-means model attributes from where they are saved as a TAR file in an S3 bucket.
# 
# You'll need to access the model by the k-means training job name, and then unzip the file into `model_algo-1`. Then you can load that file using MXNet, as before.

# In[50]:


# download and unzip the kmeans model file
kmeans_job_name = 'kmeans-2020-04-30-00-14-17-609'

model_key = os.path.join(prefix, kmeans_job_name, 'output/model.tar.gz')

# download the model file
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# In[51]:


# get the trained kmeans params using mxnet
kmeans_model_params = mx.ndarray.load('model_algo-1')

print(kmeans_model_params)


# There is only 1 set of model parameters contained within the k-means model: the cluster centroid locations in PCA-transformed, component space.
# 
# * **centroids**: The location of the centers of each cluster in component space, identified by the k-means algorithm. 
# 

# In[52]:


# get all the centroids
cluster_centroids=pd.DataFrame(kmeans_model_params[0].asnumpy())
cluster_centroids.columns=counties_transformed.columns

display(cluster_centroids)


# ### Visualizing Centroids in Component Space
# 
# You can't visualize 7-dimensional centroids in space, but you can plot a heatmap of the centroids and their location in the transformed feature space. 
# 
# This gives you insight into what characteristics define each cluster. Often with unsupervised learning, results are hard to interpret. This is one way to make use of the results of PCA + clustering techniques, together. Since you were able to examine the makeup of each PCA component, you can understand what each centroid represents in terms of the PCA components.

# In[53]:


# generate a heatmap in component space, using the seaborn library
plt.figure(figsize = (12,9))
ax = sns.heatmap(cluster_centroids.T, cmap = 'YlGnBu')
ax.set_xlabel("Cluster")
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
ax.set_title("Attribute Value by Centroid")
plt.show()


# If you've forgotten what each component corresponds to at an original-feature-level, that's okay! You can use the previously defined `display_component` function to see the feature-level makeup.

# In[69]:


# what do each of these components mean again?
# let's use the display function, from above
component_num=2
display_component(v, counties_scaled.columns.values, component_num=component_num)


# ### Natural Groupings
# 
# You can also map the cluster labels back to each individual county and examine which counties are naturally grouped together.

# In[73]:


# add a 'labels' column to the dataframe
counties_transformed['labels']=list(map(int, cluster_labels))

# sort by cluster label 0-6
sorted_counties = counties_transformed.sort_values('labels', ascending=True)
# view some pts in cluster 0
sorted_counties.head(20)


# You can also examine one of the clusters in more detail, like cluster 1, for example. A quick glance at the location of the centroid in component space (the heatmap) tells us that it has the highest value for the `comp_6` attribute. You can now see which counties fit that description.

# In[74]:


# get all counties with label == 1
cluster=counties_transformed[counties_transformed['labels']==2]
cluster.head()


# ## Final Cleanup!
# 
# * Double check that you have deleted all your endpoints.
# * I'd also suggest manually deleting your S3 bucket, models, and endpoint configurations directly from your AWS console.
# 
# You can find thorough cleanup instructions, [in the documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html).

# ---
# # Conclusion
# 
# You have just walked through a machine learning workflow for unsupervised learning, specifically, for clustering a dataset using k-means after reducing the dimensionality using PCA. By accessing the underlying models created within  SageMaker, you were able to improve the explainability of your model and draw insights from the resultant clusters. 
# 
# Using these techniques, you have been able to better understand the essential characteristics of different counties in the US and segment them into similar groups, accordingly.
