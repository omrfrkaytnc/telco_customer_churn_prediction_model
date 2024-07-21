# Telco Customer Churn Prediction

## Project Overview
This project aims to predict customer churn for a telecommunications company using various machine learning models. The goal is to identify customers who are likely to leave the service, allowing the company to take proactive measures to retain them.

## Objectives
- Data Preprocessing
- Feature Engineering
- Initial Model Selection
- Model Training and Evaluation
- Hyperparameter Optimization



## Process

### 1. Data Preprocessing
- **Handling Missing Values**: Filled missing values in the `TotalCharges` column with the median.
- **Encoding Categorical Variables**: Used one-hot encoding for categorical features like `Gender`, `InternetService`, `Contract`, and `PaymentMethod`.
- **Scaling Features**: Scaled numerical features using StandardScaler.

### 2. Feature Engineering
- **Creating New Features**: Derived new features such as `MonthlyCharges` divided by `Tenure` to capture the average monthly charge over the tenure period.
- **Feature Selection**: Selected features that are most relevant for the churn prediction using feature importance from tree-based models.

### 3. Initial Model Selection
- **Model Candidates**: Evaluated the following models with default settings:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - LightGBM
  - CatBoost

- **Criteria for Selection**: Based on initial performance metrics such as accuracy, precision, recall, and F1-score, the top five models were selected for further tuning and evaluation.

### 4. Model Training and Evaluation
- **Train-Test Split**: Split the dataset into training (70%) and testing (30%) sets.
- **Selected Models for Detailed Evaluation**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - LightGBM
  - CatBoost

- **Evaluation Metrics**:
  - **Accuracy**: Overall correctness of the model.
  - **Precision**: Correct positive predictions out of all positive predictions.
  - **Recall**: Correct positive predictions out of all actual positives.
  - **F1-Score**: Harmonic mean of precision and recall.
  - **ROC-AUC**: Area under the ROC curve, indicating the ability to distinguish between classes.

#### Results:

| Algorithm            | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 80.0%    | 73.0%     | 65.0%  | 69.0%    | 0.84    |
| Random Forest        | 82.0%    | 76.5%     | 68.0%  | 72.0%    | 0.87    |
| Gradient Boosting    | 83.5%    | 78.0%     | 70.0%  | 73.8%    | 0.88    |
| LightGBM             | 84.0%    | 79.0%     | 71.0%  | 74.8%    | 0.89    |
| CatBoost             | 83.0%    | 77.5%     | 69.5%  | 73.3%    | 0.88    |

### 5. Hyperparameter Optimization
- **Grid Search**: Exhaustive search over specified parameter values.
- **Random Search**: Randomly samples parameter values for a specified number of iterations.
- **Bayesian Optimization**: Sequential model-based optimization for efficient hyperparameter tuning.

## Conclusion
This project provides a comprehensive approach to predicting customer churn using machine learning. Among the models evaluated, LightGBM showed the highest performance with an accuracy of 84.0% and an ROC-AUC of 0.89, making it the chosen model for deployment.

---

### Türkçe Versiyon

# Telco Müşteri Terk Tahmini

## Proje Özeti
Bu proje, bir telekomünikasyon şirketi için müşteri terkini tahmin etmeyi amaçlamaktadır. Hedef, hizmeti bırakma olasılığı yüksek olan müşterileri belirleyerek şirketin onları elde tutmak için proaktif önlemler almasını sağlamaktır.

## Amaçlar
- Veri Ön İşleme
- Özellik Mühendisliği
- İlk Model Seçimi
- Model Eğitimi ve Değerlendirme
- Hiperparametre Optimizasyonu



## Süreç

### 1. Veri Ön İşleme
- **Eksik Değerleri Ele Alma**: `TotalCharges` sütunundaki eksik değerleri medyan ile doldurduk.
- **Kategorik Değişkenlerin Kodlanması**: `Gender`, `InternetService`, `Contract` ve `PaymentMethod` gibi kategorik özellikler için one-hot encoding kullandık.
- **Özellik Ölçeklendirme**: Sayısal özellikleri StandardScaler kullanarak ölçeklendirdik.

### 2. Özellik Mühendisliği
- **Yeni Özellikler Oluşturma**: `MonthlyCharges`'ı `Tenure` ile bölerek ortalama aylık ücreti yakalayan yeni özellikler türettik.
- **Özellik Seçimi**: Ağaç tabanlı modellerden özellik önemine dayalı olarak en alakalı özellikleri seçtik.

### 3. İlk Model Seçimi
- **Model Adayları**: Aşağıdaki modelleri varsayılan ayarlarla değerlendirdik:
  - Lojistik Regresyon
  - K-En Yakın Komşu (KNN)
  - Karar Ağacı
  - Rastgele Orman
  - Destek Vektör Makinesi (SVM)
  - XGBoost
  - LightGBM
  - CatBoost

- **Seçim Kriterleri**: Başlangıç performans metriklerine (doğruluk, hassasiyet, hatırlama ve F1 skoru) dayalı olarak, daha fazla ayarlama ve değerlendirme için en iyi beş model seçildi.

### 4. Model Eğitimi ve Değerlendirme
- **Eğitim-Test Ayrımı**: Veri setini eğitim (%70) ve test (%30) setlerine ayırdık.
- **Detaylı Değerlendirme için Seçilen Modeller**:
  - Lojistik Regresyon
  - Rastgele Orman
  - Gradient Boosting
  - LightGBM
  - CatBoost

- **Değerlendirme Metrikleri**:
  - **Doğruluk**: Modelin genel doğruluğu.
  - **Hassasiyet**: Tüm pozitif tahminler içindeki doğru pozitif tahminler.
  - **Hatırlama**: Tüm gerçek pozitifler içindeki doğru pozitif tahminler.
  - **F1-Skoru**: Hassasiyet ve hatırlamanın harmonik ortalaması.
  - **ROC-AUC**: Sınıflar arasındaki ayırt etme yeteneğini gösteren ROC eğrisi altındaki alan.

#### Sonuçlar:

| Algoritma            | Doğruluk | Hassasiyet | Hatırlama | F1-Skoru | ROC-AUC |
|----------------------|----------|------------|-----------|----------|---------|
| Lojistik Regresyon   | %80.0    | %73.0      | %65.0     | %69.0    | 0.84    |
| Rastgele Orman       | %82.0    | %76.5      | %68.0     | %72.0    | 0.87    |
| Gradient Boosting    | %83.5    | %78.0      | %70.0     | %73.8    | 0.88    |
| LightGBM             | %84.0    | %79.0      | %71.0     | %74.8    | 0.89    |
| CatBoost             | %83.0    | %77.5      | %69.5     | %73.3    | 0.88    |

### 5. Hiperparametre Optimizasyonu
- **Grid Search**: Belirtilen parametre değerleri üzerinde kapsamlı arama.
- **Random Search**: Belirli sayıda iterasyon için rastgele örneklenen parametre değerleri.
- **Bayesian Optimization**: Verimli hiperparametre ayarlaması için ardışık model tabanlı optimizasyon.

## Sonuç
Bu proje, makine öğrenimini kullanarak müşteri terkini tahmin etmek için kapsamlı bir yaklaşım sunmaktadır. Değerlendirilen modeller arasında, LightGBM en yüksek performansı %84.0 doğruluk ve 0.89 ROC-AUC ile göstererek dağıtım için seçilen model olmuştur.
