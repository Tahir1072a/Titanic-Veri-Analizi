# Kapsamlı Titanic Veri Analizi ve Makine Öğrenmesi Modeli

Bu proje, Kaggle'ın ünlü "Titanic: Machine Learning from Disaster" veri setini kullanarak, yolcuların hayatta kalma oranlarını etkileyen faktörleri analiz etmeyi ve bu faktörlere dayalı bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Proje, veri temizleme, özellik mühendisliği ve model optimizasyonu gibi temel veri bilimi adımlarını kapsamaktadır.

---

## 📚 Veri Seti

Bu projede kullanılan veri seti, Kaggle platformundaki **"Titanic - Machine Learning from Disaster"** yarışmasından temin edilmiştir.
* **Veri Seti Linki:** [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

---

## 🛠️ Kullanılan Teknolojiler

Projenin geliştirilmesinde aşağıdaki kütüphanelerden yararlanılmıştır:
* **Pandas:** Veri manipülasyonu, temizleme ve analizi.
* **NumPy:** Sayısal hesaplamalar ve veri dönüşümleri.
* **Matplotlib:** Veri görselleştirme ve model sonuçlarının analizi.
* **Scikit-learn:** Özellik ön işleme, modelleme ve hiperparametre optimizasyonu.

---

## 🚀 Proje İş Akışı

Proje, standart bir veri bilimi yaşam döngüsü takip edilerek tamamlanmıştır:

1.  **Veri Keşfi ve Temizleme (EDA & Cleaning):**
    * Veri seti yüklendi ve `info()`, `head()` gibi fonksiyonlarla ilk yapısal analiz gerçekleştirildi.
    * Eksik değerler tespit edildi. `Age` sütunundaki boşluklar ortalama yaş değeri ile, `Embarked` sütunundaki boşluklar ise en sık tekrar eden değer (mod) ile dolduruldu.
    * `Cabin` sütunundaki eksiklik, yolcunun bir kabini olup olmadığını belirten `Has_Cabin` adında yeni bir ikili özelliğe dönüştürüldü. Bu, veri kaybını önleyen daha anlamlı bir yaklaşımdır.

2.  **Özellik Mühendisliği (Feature Engineering):**
    * Modelleme için anlamlı olmayan `PassengerId`, `Name`, `Ticket` ve `Cabin` (dönüştürüldükten sonra) gibi sütunlar veri setinden çıkarıldı.
    * Kategorik olan `Sex` özelliği, makine öğrenmesi modelinin işleyebilmesi için sayısal forma dönüştürüldü.

3.  **Veri Ön İşleme (Preprocessing):**
    * `scikit-learn` kütüphanesinin `ColumnTransformer` ve `Pipeline` yapıları kullanılarak modern bir ön işleme hattı kuruldu.
    * **Sayısal Sütunlar:** `StandardScaler` ile ölçeklendirildi.
    * **Kategorik Sütunlar:** `OneHotEncoder` ile sayısal vektörlere dönüştürüldü.

4.  **Model Geliştirme ve Optimizasyon:**
    * Problem için **KNN**, **Decision Tree** ve **Random Forest** gibi farklı sınıflandırma modelleri denendi.
    * En iyi hiperparametreleri güvenilir bir şekilde bulmak ve veri sızıntısını önlemek amacıyla **`GridSearchCV`** ile **Çapraz Doğrulama (Cross-Validation)** yöntemi kullanılmıştır.
    * Yapılan optimizasyon sonucunda Random Forest modeli için en iyi parametreler `{'max_depth': 6, 'n_estimators': 22}` olarak belirlenmiştir. Bu parametrelerle elde edilen ortalama çapraz doğrulama doğruluğu **%83.28** olmuştur.

---

## 📊 Model Sonuçları

`GridSearchCV` ile bulunan en iyi model, daha önce hiç görmediği test verisi üzerinde değerlendirilmiştir. Elde edilen nihai sonuçlar şöyledir:

* **Test Seti Doğruluğu (Accuracy):** **%90**

### Sınıflandırma Raporu (Classification Report)

|               | Precision | Recall | F1-Score | Support |
| :------------ | :-------: | :----: | :------: | :-----: |
| **0 (Kalamadı)** |   0.89    |  0.96  |   0.93   |   266   |
| **1 (Kaldı)** |   0.92    |  0.80  |   0.86   |   152   |
| **Macro Avg** |   0.91    |  0.88  |   0.89   |   418   |
| **Weighted Avg**|   0.90    |  0.90  |   0.90   |   418   |

Bu sonuçlar, modelin özellikle hayatta kalamayanları tespit etmede çok başarılı olduğunu (%96 Recall), hayatta kalanları tahmin etme konusunda ise yüksek bir isabet oranına (%92 Precision) sahip olduğunu göstermektedir.

---

## ⚙️ Kurulum ve Çalıştırma

Proje dosyalarını yerel makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/](https://github.com/)<kullanici-adiniz>/<repo-adiniz>.git
    ```
2.  **Klasöre Gidin:**
    ```bash
    cd <repo-adiniz>
    ```
3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Jupyter Notebook'u Çalıştırın:**
    Ana analiz ve modelleme adımları `src/titanic-survive-analysis.ipynb` dosyasında bulunmaktadır.