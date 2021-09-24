import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import env
from datetime import date
from sklearn.model_selection import train_test_split

#Acquire zillow dat set

#db access
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#zillow db
zillow_sql = "SELECT *\
                FROM properties_2017\
                LEFT JOIN predictions_2017 USING(parcelid)\
                LEFT JOIN airconditioningtype USING(airconditioningtypeid)\
                LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)\
                LEFT JOIN buildingclasstype USING(buildingclasstypeid)\
                LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)\
                LEFT JOIN propertylandusetype USING(propertylandusetypeid)\
                LEFT JOIN storytype USING(storytypeid)\
                LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)\
                LEFT JOIN unique_properties USING(parcelid)\
                WHERE properties_2017.id IN(\
                SELECT DISTINCT id\
                FROM properties_2017\
                WHERE predictions_2017.transactiondate LIKE '2017%%') AND latitude IS NOT NULL;"

#acquires zillow dataset
def get_zillow_data():
    return pd.read_sql(zillow_sql,get_connection('zillow'))

def handle_missing_values(df, prop_required_column = .6, prop_required_row = .75):
#function that will drop rows or columns based on the percent of values that are missing:\
#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):
#remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df

#cleans and prepares zillow dataset
def wrangle_zillow(df):
#Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]

#Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

#Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)

#Add column for counties
    df['county'] = df['fips'].apply(
        lambda x: 'Los Angeles' if x == 6037\
        else 'Orange' if x == 6059\
        else 'Ventura')

#drop unnecessary columns
    dropcols = ['parcelid',
         'calculatedbathnbr',
         'finishedsquarefeet12',
         'fullbathcnt',
         'heatingorsystemtypeid',
         'propertycountylandusecode',
         'propertylandusetypeid',
         'propertyzoningdesc',
         'censustractandblock',
         'propertylandusedesc',
         'id']

    df = remove_columns(df, dropcols)

#replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)

#assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)

#replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

#Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 8000]

#Just to be sure we caught all nulls, drop them here
    df = df.dropna()

#get the age of the home    
    df['age'] = date.today().year - df.yearbuilt

#calculates the tax rate    
    df['tax_rate'] = (df['taxamount'] / df['taxvaluedollarcnt'])

#calculate price per sqft
    df['price_per_sqft'] = (df['taxvaluedollarcnt'] / df['calculatedfinishedsquarefeet'])

#drops the above calculations    
    df = df.drop(columns=['yearbuilt', 'taxamount', 'buildingqualitytypeid', 'fips', 'rawcensustractandblock', 'regionidcity', 'regionidcounty',
    'regionidzip', 'structuretaxvaluedollarcnt', 'assessmentyear', 'transactiondate', 'roomcnt', 'heatingorsystemdesc', 'unitcnt'])

#renames columns    
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet":"square_feet",
                        'lotsizesquarefeet':'lot_size', 'landtaxvaluedollarcnt':'property_value', "taxvaluedollarcnt":"tax_value"})

#convert dtypes
    convert_dict_int = {'bathrooms': int, 'bedrooms': int, 'square_feet':int, 'lot_size':int, 'tax_value':int, 
                    'property_value': int, 'age':int, 'price_per_sqft':int}
    df = df.astype(convert_dict_int)

    return df

#plots histogram
def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['county']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()

#Gets box plots of acquired continuous variables (non-categorical - object)
def get_box(df, cols):
    ''' Gets boxplots of acquired continuous variables'''

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()

#removes identified outliers 
def remove_outliers(df, k , col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe. Much like the word “software”, John Tukey is responsible for creating this “rule” called the 
        Inter-Quartile Range rule. In the absence of a domain knowledge reason for removing certain outliers, this is a pretty 
        robust tool for removing the most extreme outliers (with Zillow data, we can feel confident using this, since Zillow markets 
        to the majority of the bell curve and not folks w/ $20mil properties). the value for k is a constant that sets the threshold.
        Usually, you’ll see k start at 1.5, or 3 or less, depending on how many outliers you want to keep. The higher the k, the more 
        outliers you keep. Recommend not going beneath 1.5, but this is worth using, especially with data w/ extreme high/low values.'''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#split dataset
def train_validate_test_split(df):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123, stratify=df['logerror'])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123, stratify=train_validate['logerror'])
    return train, validate, test
