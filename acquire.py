import pandas as pd
import env

#db access
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#zillow db
zillow_sql = "SELECT bathroomcnt, bedroomcnt, fips, yearbuilt, taxvaluedollarcnt, taxamount, calculatedfinishedsquarefeet\
                FROM properties_2017\
                LEFT JOIN predictions_2017 USING(parcelid)\
                JOIN propertylandusetype USING(propertylandusetypeid)\
                WHERE propertylandusetype.propertylandusetypeid LIKE '261' AND predictions_2017.transactiondate BETWEEN '2017-05-01' AND '2017-08-31';"

#acquires zillow dataset
def get_zillow_data():
    return pd.read_sql(zillow_sql,get_connection('zillow'))