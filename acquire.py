import pandas as pd
import env

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