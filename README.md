# Kapsamlı Titanic Veri Analizi ve Makine Öğrenmesi Modeli

Bu proje, Kaggle'ın ünlü "Titanic: Machine Learning from Disaster" veri setini kullanarak, yolcuların hayatta kalma oranlarını etkileyen faktörleri analiz etmeyi ve bu faktörlere dayalı bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Proje, **veri temizleme**, özellik mühendisliği **(feature engineer)** ve **model optimizasyonu** gibi temel veri bilimi adımlarını kapsamaktadır.

---

## 📚 Veri Seti

Bu projede kullanılan veri seti, Kaggle platformundaki **"Titanic - Machine Learning from Disaster"** yarışmasından temin edilmiştir.
* **Veri Seti Linki:** [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

---

## 🛠️ Kullanılan Teknolojiler
Bu projenin hayata geçirilmesinde aşağıdaki veri bilimi araçlarından yararlanılmıştır:

* **Pandas:** Veri manipülasyonu, temizleme ve analizi.
* **NumPy:** Sayısal hesaplamalar ve veri dönüşümleri.
* **Matplotlib:** Veri görselleştirme ve model sonuçlarının analizi.
* **Scikit-learn:** Özellik ön işleme, modelleme ve hiperparametre optimizasyonu.

---

## 🚀 Proje İş Akışı ve Metodoloji

Proje, güvenilir ve tekrarlanabilir sonuçlar elde etmek amacıyla modern veri bilimi pratikleri izlenerek yapılandırılmıştır:

1.  **Keşifsel Veri Analizi (EDA):**

      * Veri setinin yapısı incelendi, istatistiksel özetleri çıkarıldı.
      * Cinsiyet, Bilet Sınıfı (Pclass) ve Gemiye Biniş Limanı (Embarked) gibi kategorik değişkenlerin hayatta kalma oranı üzerindeki etkileri görselleştirildi.
      * `Sex` ve `Pclass`'ın hayatta kalma üzerinde en güçlü etkiye sahip olduğu tespit edildi.

2.  **Veri Temizleme ve Özellik Mühendisliği:**

      * **Yaş (`Age`) Doldurma:** Eksik yaş verileri, tüm veri setinin genel ortalaması gibi basit bir yöntem yerine, her yolcunun ait olduğu **sosyo-demografik grubun (Cinsiyet ve Bilet Sınıfına göre)** medyan yaşıyla doldurularak daha isabetli ve bağlama uygun bir tamamlama yapıldı.
      * **Yeni Özellikler Türetme:**
          * `Has_Cabin`: `Cabin` sütunundaki eksiklik, yolcunun bir kabini olup olmadığını belirten yeni bir ikili özelliğe dönüştürüldü.
          * `Title`: `Name` sütunundan 'Mr', 'Mrs', 'Miss' gibi unvanlar çıkarılarak sosyal statüyü temsil eden yeni bir kategorik özellik yaratıldı.
          * `FamilySize`: `SibSp` ve `Parch` sütunları birleştirilerek toplam aile boyutu hesaplandı.
      * **Gereksiz Sütunların Çıkarılması:** Modelleme için anlamlı olmayan `PassengerId`, `Name`, `Ticket` gibi sütunlar veri setinden çıkarıldı.

3.   **Model Geliştirme ve Optimizasyon:**

      * `KNN`, `Decision Tree` ve `Random Forest` gibi farklı sınıflandırma modellerinin performansları karşılaştırıldı.
      * Her model için en iyi hiperparametreler, `GridSearchCV` ve 10-katmanlı `StratifiedKFold` Çapraz Doğrulama yöntemi kullanılarak bulundu.
      * Yapılan optimizasyon sonucunda **Random Forest** modeli en yüksek çapraz doğrulama skorunu verdi. En iyi parametreler `{'max_depth': 6, 'n_estimators': 22}` olarak belirlendi ve bu parametrelerle elde edilen ortalama doğruluk **%83.28** oldu.

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