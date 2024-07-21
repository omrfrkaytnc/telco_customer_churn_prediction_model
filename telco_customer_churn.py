
##################################
# Telco Customer Churn Prediction
##################################
# Problem: It is expected to develop a machine learning model that can predict customers who will leave the company.
# Before developing the model, necessary data analysis and feature engineering steps are expected to be performed.
# The Telco customer churn data contains information about a fictitious telecommunications company providing landline and
# Internet services to 7043 customers in California in the third quarter. It includes information about which customers
# have left the service, stayed, or signed up for the service.
# 21 Variables 7043 Observations


# Problem : Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.
# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.
# 21 Değişken 7043 Gözlem

# CustomerId : Customer ID / Müşteri İd’si
# Gender : Gender /Cinsiyet
# SeniorCitizen : Whether the customer is a senior citizen (1, 0) / Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Whether the customer has a partner (Yes, No) - marital status / Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Whether the customer has dependents (Yes, No) (e.g., children, parents, grandparents) / Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# Tenure : Number of months the customer has been with the company / Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Whether the customer has phone service (Yes, No) / Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Whether the customer has multiple lines (Yes, No, No phone service) / Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : The customer's Internet service provider (DSL, Fiber optic, No) / Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity : Whether the customer has online security (Yes, No, No Internet service) / Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Whether the customer has online backup (Yes, No, No Internet service) / Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Whether the customer has device protection (Yes, No, No Internet service) / Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Whether the customer receives tech support (Yes, No, No Internet service) / Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Whether the customer has TV streaming (Yes, No, No Internet service) - Indicates whether the customer uses Internet service to stream TV programs from a third-party provider / Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies : Whether the customer has movie streaming (Yes, No, No Internet service) - Indicates whether the customer uses Internet service to stream movies from a third-party provider / Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract : The customer's contract term (Month-to-month, One year, Two years) / Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Whether the customer has paperless billing (Yes, No) / Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) / Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : The amount billed to the customer monthly / Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : The total amount billed to the customer / Müşteriden tahsil edilen toplam tutar
# Churn : Whether the customer has churned (Yes or No) - Indicates whether the customer left within the last month or quarter / Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler


# Each row represents a unique customer.
# The variables contain information about customer service, account, and demographic data.
# Services customers have subscribed to include phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
# Customer account information includes how long they have been a customer, contract type, payment method, paperless billing, monthly charges, and total charges.
# Demographic information about customers includes gender, age range, and whether they have partners or dependents.

# Her satır benzersiz bir müşteriyi temsil etmekte.
# Değişkenler müşteri hizmetleri, hesap ve demografik veriler hakkında bilgiler içerir.
# Müşterilerin kaydolduğu hizmetler - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Müşteri hesap bilgileri – ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler ve toplam ücretler
# Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı ve ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı






# Import Necessary Libraries and Functions / Gerekli Kütüphane ve Fonksiyonların Kurulumu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

# Set Display Options / Görüntüleme Seçeneklerini Ayarla
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load Dataset / Veri Kümesini Yükle
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df.head()   # Display first few rows / İlk birkaç satırı görüntüle
df.shape # (7043, 21) # Show the shape of the dataframe / Veri çerçevesinin şeklini göster
df.info()  # Display information about the dataframe / Veri çerçevesi hakkında bilgi göster

# Convert TotalCharges to Numeric / TotalCharges'i Sayısal Veriye Dönüştür
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Encode Churn as Binary / Churn'u İkili Koda Dönüştür
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

df.info()  # Display updated information about the dataframe / Güncellenmiş veri çerçevesi bilgilerini göster


##################################
# TASK 1: EXPLORATORY DATA ANALYSIS / GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENERAL OVERVIEW / GENEL RESİM
##################################

def check_df(dataframe, head=5):
    """
    This function performs an exploratory data analysis on the provided DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to analyze.
    head (int): Number of rows to display for head and tail. Default is 5.

    Bu fonksiyon verilen DataFrame üzerinde keşifsel veri analizi yapar.

    Parametreler:
    dataframe (pd.DataFrame): Analiz edilecek DataFrame.
    head (int): Baş ve son için görüntülenecek satır sayısı. Varsayılan 5'tir.
    """
    # Print the shape of the DataFrame / DataFrame'in boyutunu yazdır
    print("##################### Shape #####################")
    print(dataframe.shape)

    # Print the data types of each column / Her sütunun veri tiplerini yazdır
    print("##################### Types #####################")
    print(dataframe.dtypes)

    # Print the first few rows of the DataFrame / DataFrame'in ilk birkaç satırını yazdır
    print("##################### Head #####################")
    print(dataframe.head(head))

    # Print the last few rows of the DataFrame / DataFrame'in son birkaç satırını yazdır
    print("##################### Tail #####################")
    print(dataframe.tail(head))

    # Print the number of missing values in each column / Her sütundaki eksik değer sayısını yazdır
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    # Print quantile statistics for numerical columns / Sayısal sütunlar için kantil istatistiklerini yazdır
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# Call the function to check the DataFrame / DataFrame'i kontrol etmek için fonksiyonu çağır
check_df(df)


##################################
# CATCHING NUMERIC AND CATEGORICAL VARIABLES / NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Retrieves the names of categorical, numeric, and categorical but cardinal variables in the dataset.
    Note: Categorical variables include numeric-looking categorical variables as well.

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                The threshold for categorical variables that look numeric
        car_th: int, optional
                The threshold for cardinal categorical variables

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be extracted
        cat_th: int, optional
                Class threshold for numeric-looking categorical variables
        car_th: int, optional
                Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numeric variables
        cat_but_car: list
                List of categorical-looking cardinal variables

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is included in cat_cols.

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


##################################
# CATEGORICAL VARIABLES ANALYSIS / KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    # Print value counts and their percentages / Değer sayıları ve yüzdelerini yazdır
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    # Optionally plot a count plot / İsteğe bağlı olarak bir count plot çizin
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# Apply the function to each categorical column / Fonksiyonu her kategorik sütuna uygulayın
for col in cat_cols:
    cat_summary(df, col, plot=True)


##################################
# NUMERIC VARIABLES ANALYSIS / NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    # Print descriptive statistics and quantiles / Tanımlayıcı istatistikler ve kuantilleri yazdır
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    # Optionally plot a histogram / İsteğe bağlı olarak bir histogram çizin
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

# Apply the function to each numerical column / Fonksiyonu her sayısal sütuna uygulayın
for col in num_cols:
    num_summary(df, col, plot=True)




##################################
# ANALYSIS OF NUMERICAL VARIABLES BY TARGET VARIABLE / HEDEF DEĞİŞKENE GÖRE NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    # Print the mean of the numerical column grouped by the target variable / Hedef değişkenine göre gruplanmış numerik sütunun ortalamasını yazdır
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

# Apply the function to each numerical column / Fonksiyonu her numerik sütuna uygulayın
for col in num_cols:
    target_summary_with_num(df, "Churn", col)




##################################
# ANALYSIS OF CATEGORICAL VARIABLES BY TARGET / KATEGORİK DEĞİŞKENLERİN HEDEF DEĞİŞKENE GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    # Print the categorical column name / Kategorik sütun adını yazdır
    print(categorical_col)
    # Print the mean of the target variable grouped by the categorical column,
    # along with counts and ratios of each category / Kategorik sütuna göre gruplanmış hedef değişkeninin ortalamasını,
    # her kategorinin sayısını ve oranlarını yazdır
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

# Apply the function to each categorical column / Fonksiyonu her kategorik sütuna uygulayın
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Display the DataFrame info / DataFrame'in genel bilgilerini göster
df.info()



##################################
# CORRELATION / KORELASYON
##################################

# Calculate correlation matrix / Korelasyon matrisini hesapla
df[num_cols].corr()

# Correlation Matrix Plot / Korelasyon Matrisi Grafiği
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte / TotalChargers is highly correlated with monthly charges and tenure

# Let's check the variables with the highest correlation to Churn / Churn ile en yüksek korelasyona sahip değişkenlere bakalım:
df.corrwith(df["Churn"]).sort_values(ascending=False)

# We observe that as the monthly payment and customer age increase, churn increases as well! / Aylık ödeme ve müşteri yaşı arttıkça churn'ün arttığını görüyoruz!



##################################
# FEATURE ENGINEERING / ÖZELLİK MÜHENDİSLİĞİ
##################################

##################################
# MISSING VALUE ANALYSIS / EKSİK DEĞER ANALİZİ
##################################

# Check for missing values / Eksik değerleri kontrol et
df.isnull().sum()
# customerID           0
# gender               0
# SeniorCitizen        0
# Partner              0
# Dependents           0
# tenure               0
# PhoneService         0
# MultipleLines        0
# InternetService      0
# OnlineSecurity       0
# OnlineBackup         0
# DeviceProtection     0
# TechSupport          0
# StreamingTV          0
# StreamingMovies      0
# Contract             0
# PaperlessBilling     0
# PaymentMethod        0
# MonthlyCharges       0
# TotalCharges        11
# Churn                0
# dtype: int64
def missing_values_table(dataframe, na_name=False):
    # Find columns with missing values / Eksik değer bulunan sütunları bul
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

# Get columns with missing values / Eksik değer bulunan sütunları al
na_columns = missing_values_table(df, na_name=True)
#               n_miss  ratio
# TotalCharges      11  0.160

# Fill missing values in 'TotalCharges' with the median value / 'TotalCharges' sütunundaki eksik değerleri medyan değerle doldur
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)  # Monthly charges may be used to fill 'TotalCharges' (try if better) or 11 variables can be dropped

# Verify that missing values have been handled / Eksik değerlerin işlendiğini doğrula
df.isnull().sum()
# customerID          0
# gender              0
# SeniorCitizen       0
# Partner             0
# Dependents          0
# tenure              0
# PhoneService        0
# MultipleLines       0
# InternetService     0
# OnlineSecurity      0
# OnlineBackup        0
# DeviceProtection    0
# TechSupport         0
# StreamingTV         0
# StreamingMovies     0
# Contract            0
# PaperlessBilling    0
# PaymentMethod       0
# MonthlyCharges      0
# TotalCharges        0
# Churn               0
# dtype: int64



##################################
# BASE MODEL SETUP / BASE MODEL KURULUMU
##################################

# Make a copy of the dataframe / Veri çerçevesinin bir kopyasını oluştur
dff = df.copy()

# Exclude the target column from categorical columns / Kategorik sütunlardan hedef sütunu hariç tut
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # Apply one-hot encoding to the categorical columns / Kategorik sütunlara one-hot encoding uygula
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Apply one-hot encoding to the dataframe / Veri çerçevesine one-hot encoding uygula
dff = one_hot_encoder(dff, cat_cols, drop_first=True)
dff.shape

# Define the target and feature columns / Hedef ve özellik sütunlarını tanımla
y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)

# List of models to evaluate / Değerlendirilecek modeller listesi
models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

# Evaluate each model using cross-validation / Her modeli çapraz doğrulama kullanarak değerlendir
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

"""
########## LR ##########
Accuracy: 0.8032
Auc: 0.8428
Recall: 0.5436
Precision: 0.6561
F1: 0.5942
########## KNN ##########
Accuracy: 0.7639
Auc: 0.7469
Recall: 0.4468
Precision: 0.5714
F1: 0.5009
########## CART ##########
Accuracy: 0.7277
Auc: 0.6579
Recall: 0.504
Precision: 0.4876
F1: 0.4954
########## RF ##########
Accuracy: 0.792
Auc: 0.8254
Recall: 0.4837
Precision: 0.6451
F1: 0.5526
########## SVM ##########
Accuracy: 0.7696
Auc: 0.7141
Recall: 0.2905
Precision: 0.6495
F1: 0.4009
########## XGB ##########
Accuracy: 0.783
Auc: 0.8243
Recall: 0.5126
Precision: 0.6107
F1: 0.5568
########## LightGBM ##########
Accuracy: 0.7967
Auc: 0.8361
Recall: 0.5297
Precision: 0.6437
F1: 0.5805
########## CatBoost ##########
Accuracy: 0.8001
Auc: 0.8413
Recall: 0.5131
Precision: 0.6595
F1: 0.5767
"""


##################################
# OUTLIER ANALYSIS / AYKIRI DEĞER ANALİZİ
##################################

# Display basic statistics of numerical columns / Sayısal sütunların temel istatistiklerini göster
df.describe().T

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    # Calculate the lower and upper thresholds for outliers / Aykırı değerler için alt ve üst eşikleri hesapla
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    # Check if there are outliers in the column / Sütunda aykırı değer olup olmadığını kontrol et
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    # Replace outliers with the defined thresholds / Aykırı değerleri tanımlanan eşiklerle değiştir
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Outlier Analysis and Capping / Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)





##################################
# FEATURE EXTRACTION / ÖZELLİK ÇIKARIMI
##################################

# Creating a categorical variable from the tenure variable / Tenure değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

df.head()
df.shape # (7043, 22)

# Labeling customers with 1 or 2-year contracts as Engaged / Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# People without any backup, protection, or support / Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Monthly contract holders who are young / Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Total number of services a customer has / Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Flag for customers with any streaming service / Herhangi bir streaming hizmeti alan kişiler için flag
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Flag for customers using automatic payment methods / Kişi otomatik ödeme yapıyor mu? için flag
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# Average monthly payment / Ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Increase of current price compared to the average price / Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Service fee per service / Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

df.head()
df.shape  # (7043, 31)
print(df['Partner'].dtype)
print(df['Dependents'].dtype)
df['Partner'] = pd.to_numeric(df['Partner'], errors='coerce')
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

# Customer's solo status / Müşterinin yalnız olma durumu
df.loc[((df['Partner'] + df['Dependents']) > 0), "new_is_alone"] = "NO"
df.loc[((df['Partner'] + df['Dependents']) == 0), "new_is_alone"] = "YES"

# Internet and phone services / İnternet ve telefon servisleri
df.loc[(df["PhoneService"] == 1) & (df["InternetService"] != "No"), "new_services"] = "phone & internet"
df.loc[(df["PhoneService"] == 0) & (df["InternetService"] != "No"), "new_services"] = "internet"
df.loc[(df["PhoneService"] == 1) & (df["InternetService"] == "No"), "new_services"] = "phone"

# Converting the tenure variable into categorical / Tenure değişkenini kategorik hale getirme
df["new_tenure_cat"] = pd.qcut(df["tenure"], q=3, labels=["short_term", "mid_term", "long_term"])





##################################
# ENCODING
##################################

# Separating variables by their types / Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# Identifying binary columns (columns with exactly two unique values) / İkili sütunları belirleme (tam olarak iki benzersiz değere sahip sütunlar)
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

# Applying label encoding to binary columns / İkili sütunlara label encoding uygulama
for col in binary_cols:
    df = label_encoder(df, col)

# Checking the updated dataframe / Güncellenmiş dataframe'i kontrol etme
df.head()

# One-Hot Encoding Process / One-Hot Encoding İşlemi
# Updating the cat_cols list to exclude binary columns and certain other columns / cat_cols listesini ikili sütunlar ve bazı diğer sütunları hariç tutacak şekilde güncelleme
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Applying one-hot encoding to the categorical columns / Kategorik sütunlara one-hot encoding uygulama
df = one_hot_encoder(df, cat_cols, drop_first=True)

# Checking the updated dataframe / Güncellenmiş dataframe'i kontrol etme
df.head()
df.shape



##################################
# MODELING / MODELLEME
##################################

# Defining the target variable and features / Hedef değişkeni ve özellikleri tanımlama
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

# Defining the models to be evaluated / Değerlendirilecek modelleri tanımlama
models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

# Evaluating each model using cross-validation / Her bir modeli çapraz doğrulama kullanarak değerlendirme
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")  # Accuracy (Doğruluk)
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")        # AUC (ROC Eğrisi Altındaki Alan)
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")      # Recall (Duyarlılık)
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")# Precision (Kesinlik)
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")              # F1 Score (F1 Skoru)

"""
########## LR ##########
Accuracy: 0.8026
Auc: 0.8439
Recall: 0.5078
Precision: 0.6698
F1: 0.577
########## KNN ##########
Accuracy: 0.7691
Auc: 0.7541
Recall: 0.4628
Precision: 0.5829
F1: 0.5152
########## CART ##########
Accuracy: 0.723
Auc: 0.652
Recall: 0.4955
Precision: 0.4788
F1: 0.4867
########## RF ##########
Accuracy: 0.7944
Auc: 0.8273
Recall: 0.503
Precision: 0.6454
F1: 0.565
########## SVM ##########
Accuracy: 0.768
Auc: 0.7256
Recall: 0.2595
Precision: 0.6638
F1: 0.3721
########## XGB ##########
Accuracy: 0.784
Auc: 0.8238
Recall: 0.5174
Precision: 0.6104
F1: 0.5598
########## LightGBM ##########
Accuracy: 0.7931
Auc: 0.8368
Recall: 0.5206
Precision: 0.6351
F1: 0.572
########## CatBoost ##########
Accuracy: 0.7984
Auc: 0.843
Recall: 0.5211
Precision: 0.6503
F1: 0.5785
"""



###########################################
# FINAL MODELS / FİNAL MODELLER
###########################################

################################################
# Random Forests / Rastgele Ormanlar
################################################

# Define and train the Random Forest model / Rastgele Orman modelini tanımlama ve eğitme
rf_model = RandomForestClassifier(random_state=17)

# Define parameter grid for hyperparameter tuning / Hiperparametre ayarı için parametre ızgarasını tanımlama
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

# Perform GridSearch to find the best parameters / En iyi parametreleri bulmak için GridSearch uygulama
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Print the best parameters and score / En iyi parametreleri ve skoru yazdırma
print(f"Best parameters: {rf_best_grid.best_params_}")  # Best parameters: {'max_depth': 8, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 200}
print(f"Best score: {rf_best_grid.best_score_}")  # Best score: 0.801788623459578

# Train the final model with the best parameters / En iyi parametrelerle son modeli eğitme
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation / Son modeli çapraz doğrulama ile değerlendirme
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")  # Doğruluk # Accuracy: 0.8007906189555125
print(f"F1 Score: {cv_results['test_f1'].mean()}")       # F1 Score: 0.5752643932912982
print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")   # ROC AUC: 0.8459088893100983




################################################
# XGBoost
################################################

# Define and train the XGBoost model / XGBoost modelini tanımlama ve eğitme
xgboost_model = XGBClassifier(random_state=17)

# Define parameter grid for hyperparameter tuning / Hiperparametre ayarı için parametre ızgarasını tanımlama
xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

# Perform GridSearch to find the best parameters / En iyi parametreleri bulmak için GridSearch uygulama
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Print the best parameters and score / En iyi parametreleri ve skoru yazdırma
print(f"Best parameters: {xgboost_best_grid.best_params_}")  # Best parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}
print(f"Best score: {xgboost_best_grid.best_score_}") # Best score: 0.8027825383895735

# Train the final model with the best parameters / En iyi parametrelerle son modeli eğitme
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation / Son modeli çapraz doğrulama ile değerlendirme
cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")  # Accuracy: 0.8012
print(f"F1 Score: {cv_results['test_f1'].mean()}")       # F1 Score: 0.5822
print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")   # ROC AUC: 0.8454



################################################
# LightGBM
################################################

# Define and train the LightGBM model / LightGBM modelini tanımlama ve eğitme
lgbm_model = LGBMClassifier(random_state=17)

# Define parameter grid for hyperparameter tuning / Hiperparametre ayarı için parametre ızgarasını tanımlama
lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

# Perform GridSearch to find the best parameters / En iyi parametreleri bulmak için GridSearch uygulama
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Print the best parameters and score / En iyi parametreleri ve skoru yazdırma
print(f"Best parameters: {lgbm_best_grid.best_params_}")  # Best parameters: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 500}
print(f"Best score: {lgbm_best_grid.best_score_}") # Best score: 0.8009

# Train the final model with the best parameters / En iyi parametrelerle son modeli eğitme
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation / Son modeli çapraz doğrulama ile değerlendirme
cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")  # Accuracy: 0.8040
print(f"F1 Score: {cv_results['test_f1'].mean()}")       # F1 Score: 0.5880
print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")   # ROC AUC: 0.8457



################################################
# CatBoost
################################################

# Define and train the CatBoost model / CatBoost modelini tanımlama ve eğitme
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

# Define parameter grid for hyperparameter tuning / Hiperparametre ayarı için parametre ızgarasını tanımlama
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

# Perform GridSearch to find the best parameters / En iyi parametreleri bulmak için GridSearch uygulama
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Print the best parameters and score / En iyi parametreleri ve skoru yazdırma
print(f"Best parameters: {catboost_best_grid.best_params_}")  # Best parameters: {'depth': 3, 'iterations': 500, 'learning_rate': 0.01}
print(f"Best score: {catboost_best_grid.best_score_}") # Best score: 0.8039

# Train the final model with the best parameters / En iyi parametrelerle son modeli eğitme
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation / Son modeli çapraz doğrulama ile değerlendirme
cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")  # Accuracy: 0.8032
print(f"F1 Score: {cv_results['test_f1'].mean()}")       # F1 Score: 0.5712
print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")   # ROC AUC: 0.8477


################################################
# Feature Importance / Özelliklerin Önem Dereceleri
################################################

def plot_importance(model, features, num=len(X), save=False):
    """
    Plot and optionally save the feature importances of the model.

    Parameters:
    model (object): Trained model with feature_importances_ attribute.
    features (DataFrame): DataFrame containing the feature names.
    num (int): Number of top features to display.
    save (bool): Whether to save the plot as an image.
    ----
    Modelin özellik önem derecelerini çiz ve isteğe bağlı olarak kaydet.
    
    Parametreler:
    model (object): Özelliklerin önem derecelerini içeren eğitilmiş model.
    features (DataFrame): Özellik isimlerini içeren DataFrame.
    num (int): Görüntülenecek en iyi özellik sayısı.
    save (bool): Grafiği bir resim olarak kaydetmek isteyip istemediğiniz.
    """
    # Create a DataFrame for feature importances / Özelliklerin önem derecelerini içeren bir DataFrame oluşturma
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})

    # Plotting / Çizim
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    # Optionally save the plot / İsteğe bağlı olarak grafiği kaydetme
    if save:
        plt.savefig('importances.png')


# Plot feature importances for different models / Farklı modeller için özelliklerin önem derecelerini çiz
plot_importance(rf_final, X)  # Plot for Random Forest /  Random Forest için çizim
plot_importance(xgboost_final, X)  # Plot for XGBoost /  XGBoost için çizim
plot_importance(lgbm_final, X)  # Plot for LightGBM /  LightGBM için çizim
plot_importance(catboost_final, X)  # Plot for CatBoost /  CatBoost için çizim

