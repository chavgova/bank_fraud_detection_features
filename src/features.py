import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import phonenumbers
import pycountry

def get_encoded_features(data: pd.Series) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False).fit(data.to_numpy().reshape(-1, 1))
    encoded = encoder.transform(data.to_numpy().reshape(-1, 1))
    return pd.DataFrame(encoded, index=data.index, columns=encoder.get_feature_names_out())


def get_z_score(data: pd.DataFrame) -> pd.Series:
    return (data["amount"] - data["amount"].mean()) / data["amount"].std()

def feature_number_transactions_per_category(transaction_df: pd.DataFrame, reference_dates: pd.DatetimeIndex, 
                                             categories: tuple[str, ...], month_windows: tuple[int, ...])-> pd.DataFrame:

    # this also makes missing dates to NaT
    transaction_dates =pd.to_datetime(transaction_df["chck_date"], format="mixed", errors="coerce", utc=True)   

    reference_dates = pd.DatetimeIndex(pd.to_datetime(reference_dates, utc=True) ).unique().sort_values()  

    feature_columns = [
        f"transaction_count_{category}_{months}_months"
        for months in month_windows
        for category in categories]

    records=[]

    for reference_date in reference_dates: 
        record={} 
        # I have decided to consider a month window to be exaxtly a calendar month  
        # so a stransaction on 10th of march will in the pastwindow of 10th of april 
        # this way the feature is more robust, interpretable and transaction-specific
        for months in month_windows:
            cutoff_date = reference_date - pd.DateOffset(months=months) 

            # we include transactions before the reference transaction date 
            in_window = ((transaction_dates >= cutoff_date) & (transaction_dates < reference_date))  

            category_counts = (transaction_df.loc[in_window].groupby("category", observed=True).size())  

            for category in categories:
                column_name = (f"transaction_count_{category}_{months}_months")
                record[column_name] = category_counts.get(category, 0) # no transactions per cat => 0

        records.append(record)

    number_of_transactions_per_category_df = pd.DataFrame(records, index=reference_dates, columns=feature_columns, dtype="int64").rename_axis("reference_date")
     
    return number_of_transactions_per_category_df  

def clean_phone_number(phone_str: str) -> str:
    phone_str= str(phone_str).strip()  
    phone_str= phone_str.replace(" ","").replace("-","") 
    phone_str= phone_str.replace("(","").replace(")","")
    
    if phone_str.startswith("00"): 
        phone_str = "+" + phone_str[2:]   
    if not phone_str.startswith("+"):  
        return None 
    
    return phone_str

def parse_phone_to_country(phone_str: str) -> str:
    phone_str = clean_phone_number(phone_str)
    if not phone_str:
        return None
    
    try:
        parsed_phone= phonenumbers.parse(phone_str, None)
        country_code= phonenumbers.region_code_for_number(parsed_phone)
        
        if country_code not in phonenumbers.SUPPORTED_REGIONS:
            return None # making sure an unsupported code does not register as known and different
        return country_code 
    except:
        return None

def parse_iban_to_country(iban_str: str) -> str:
    try: 
        country_code = iban_str.strip()[:2].upper()  
        country = pycountry.countries.get(alpha_2=country_code) 
        return country_code if country else None  
    except: 
        return None

def feature_transaction_made_in_phone_registered_country(transaction_df: pd.DataFrame) -> pd.Series:
    transaction_country = transaction_df["cust_acc"].map(parse_iban_to_country)
    phone_country = transaction_df["phone"].map(parse_phone_to_country)
    
    known_countries = (transaction_country.notna() & phone_country.notna())
    
    # returning 1 if both contries are known and equal
    # returning 0 if both countries are known and different
    # returning NaN if either country is unknown
     
    return (
        transaction_country.eq(phone_country)
        .astype("float64")
        .where(known_countries)
        .rename("transaction_made_in_phone_registered_country"))
