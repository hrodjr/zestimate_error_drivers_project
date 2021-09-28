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

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
#function that will drop rows or columns based on the percent of values that are missing:\
#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def summarize(df):
    print(df.shape)
    print(f'___________________________')
    print(df.info())
    print(f'___________________________')      
    print(df.isnull().sum())

    

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
                'propertylandusetypeid',
                'heatingorsystemtypeid', 
                'id', 
                'buildingqualitytypeid', 
                'calculatedbathnbr',
                'finishedsquarefeet12', 
                'fullbathcnt', 
                'propertycountylandusecode', 
                'propertyzoningdesc', 
                'rawcensustractandblock',
                'regionidcity', 
                'regionidcounty', 
                'regionidzip', 
                'roomcnt', 
                'unitcnt', 
                'structuretaxvaluedollarcnt', 
                'assessmentyear',
                'censustractandblock', 
                'transactiondate', 
                'heatingorsystemdesc', 
                'propertylandusedesc', 
                'landtaxvaluedollarcnt']

    df = remove_columns(df, dropcols)

#get the age of the home    
    df['age'] = date.today().year - df.yearbuilt

#calculates the tax rate    
    df['tax_rate'] = (df['taxamount'] / df['taxvaluedollarcnt'])

#calculate price per sqft
    df['price_per_sqft'] = (df['taxvaluedollarcnt'] / df['calculatedfinishedsquarefeet'])

#drop calculated columns
    df = df.drop(columns = ['yearbuilt', 'taxamount'])

#fillna with means
    df.calculatedfinishedsquarefeet.fillna(1784.94, inplace = True)
    df.lotsizesquarefeet.fillna(29973.79, inplace = True)
    df.taxvaluedollarcnt.fillna(490055.43, inplace = True)
    df.age.fillna(52, inplace = True)
    df.tax_rate.fillna(0.0132, inplace = True)
    df.price_per_sqft.fillna(266.26, inplace = True)

#convert flosts to integers
    convert_dict_int = {'bathroomcnt': int, 'bedroomcnt': int, 'calculatedfinishedsquarefeet':int, 'lotsizesquarefeet':int,
                    'taxvaluedollarcnt':int, 'age': int, 'price_per_sqft':int, 'fips':int}
    df = df.astype(convert_dict_int)

#rename columns
    df = df.rename(columns={'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 'calculatedfinishedsquarefeet':'sqft',
                            'lotsizesquarefeet':'lot_size', 'taxvaluedollarcnt':'tax_value', 'fips':'county_code'})
#converts long/lat into usable location plots
    df['latitude'] = df.latitude*.000001
    df['longitude'] = df.longitude*.000001
#adds an absolute logerror column to the df
    df = absolute_logerror(df)

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
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''

# List of columns
    cols = ['bathrooms', 'bedrooms', 'sqft', 'latitude', 'longitude', 'lot_size', 'tax_value', 'age', 
            'tax_rate', 'price_per_sqft']
    
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
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)
    return train, validate, test

def absolute_logerror(df):
    '''
    This function takes in the dataframe and returns the df with new column abs_logerror
    '''
    df['abs_logerror'] = df['logerror'].abs()
    
    return df
