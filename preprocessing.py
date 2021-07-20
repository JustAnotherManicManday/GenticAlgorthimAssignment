import pandas as pd

# Read in the data (must be in working directory and changed from xlsx to csv file type  (pd.read_excel()
# doesn't work on my machine)
df = pd.read_csv('features_by_participant.csv')

# extract labels column and change to 0 or 1 (for pytorch)
labels = df.pop('label')
labels[0:] = labels[0:]-1
# drop the first column (participant number)
df.drop(df.columns[0], axis=1, inplace=True)

# Remove any columns that are all zeros
for i in range(df.shape[1]):
    if sum(df.iloc[:, i]!=0)== 0: # If the sum of non-zero elements is 0
        df.drop(df.columns[i], axis=1, inplace=True)

# Put all data on standardised z-score scale
for i in range(df.shape[1]):
    # get a column
    x = df.iloc[:, i]
    # put that column on z-score scale
    x = (x - x.mean()) / x.std()
    df.iloc[:, i] = x


# replace outliers with the mean value
count = 0
# Along rows
for i in range(df.shape[0]):
    # Along columns
    for j in range(df.shape[1]):
        # If something is 3 sd's above or below the mean change its value to the mean
        if df.iloc[i, j] >= 3 or df.iloc[i, j] <= -3:
            count += 1
            df.iloc[i, j] = df[df.columns[j]].mean()

print('replaced {} outliers with the mean value'.format(count))

# ************* Drop the features that make prediction very simple ********
# In this dataset there are some features that simplify the problem, so we're going
# to ignore them to give us some room to work in.
to_drop = ['sum_AF3', 'hjorth_P7', 'sum_P7',
           'rms_F8', 'mean_P8', 'sum_P8', 'sum_T7', 'sum_F8',
           'mean_F8', 'sum_O1', 'sum_FC6', 'sum_O2', 'mean_FC5',
           'sum_FC5', 'sum_AF4', 'rms_F7', 'sum_F7']
for col in to_drop:
    df.drop(col, axis=1, inplace=True)
    print("dropped {}".format(col))

# save the data to the working directory
df.to_csv('features_cleaned_scaled.csv', index=False)
labels.to_csv('labels.csv', index=False)

