# Kaggle Titanic Veri Seti ile Kapsamlı Veri Analizi

Bu proje, ünlü Titanic gemi kazası veri setini kullanarak yolcuların hayatta kalma oranlarını etkileyen faktörleri **NumPy** ve **Pandas** kütüphaneleriyle analiz etmektedir. Projenin amacı, veri biliminin temel adımlarını (veri yükleme, temizleme, özellik mühendisliği ve analiz) uygulayarak anlamlı içgörüler çıkarmak ve bu kütüphanelerdeki yetkinliği göstermektir.

---

## 📚 Veri Seti

Bu projede kullanılan veri seti, Kaggle platformundaki **"Titanic - Machine Learning from Disaster"** yarışmasından temin edilmiştir. Veri setine şu adresten ulaşabilirsiniz:
[https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

---

## 🛠️ Kullanılan Kütüphaneler

* **Pandas:** Veri yükleme, temizleme, manipülasyon ve analiz için temel kütüphane.
* **NumPy:** Sayısal hesaplamalar ve özellikle 'Özellik Mühendisliği' aşamasında bazı koşullu işlemler için kullanılmıştır.

---

## 🚀 Proje Adımları

Proje, aşağıdaki temel veri bilimi adımlarını takip etmiştir:

1.  **Veri Yükleme ve İlk Keşif:**
    * `train.csv` dosyası Pandas DataFrame olarak yüklendi.
    * `head()`, `tail()` ve `info()` metodlarıyla veriye ilk bakış atıldı ve eksik veriler tespit edildi (`Age`, `Cabin`, `Embarked`).

2.  **Veri Temizleme:**
    * `Age` sütunundaki eksik değerler, sütunun **ortalama** değeri ile dolduruldu (`fillna()`).
    * `Cabin` sütunundaki çok sayıda eksik değer, 'U' (Unknown - Bilinmeyen) olarak etiketlendi (`fillna()`).
    * `Embarked` sütunundaki (2 adet) eksik değer, en sık görünen biniş limanı (**mod**) ile dolduruldu (`fillna()`, `mode()`).

3.  **Özellik Mühendisliği:**
    * `SibSp` ve `Parch` sütunları kullanılarak `FamilySize` (Aile Büyüklüğü) adında yeni bir özellik oluşturuldu.
    * `FamilySize` kullanılarak `IsAlone` (Yalnız mı?) adında (1/0) yeni bir özellik oluşturuldu (`np.where`).
    * `Sex` sütunu sayısal değerlere (0: Erkek, 1: Kadın) çevrildi (`np.where`).
    * `Embarked` sütunu sayısal değerlere (S:0, C:1, Q:2) çevrildi (`map`).

4.  **Analiz ve İçgörü Çıkarma:**
    * `groupby()` metodu kullanılarak 'Sex', 'Pclass' ve 'Embarked' gibi kategorilere göre hayatta kalma oranları (`Survived` ortalaması) hesaplandı.
    * `corr()` metodu ile sayısal özellikler arasındaki (özellikle 'Survived' ile olan) korelasyonlar incelendi.

---

## 📊 Öne Çıkan Bulgular

* **Cinsiyet Etkisi:** Kadınların (`Sex=1`) hayatta kalma oranı (%74.2), erkeklerin (`Sex=0`) hayatta kalma oranından (%18.9) anlamlı derecede yüksektir. Bu, "önce kadınlar ve çocuklar" yaklaşımını desteklemektedir.
* **Sınıf Etkisi:** 1. sınıf (`Pclass=1`) yolcuların hayatta kalma oranı (%63.0) en yüksek iken, 3. sınıf (`Pclass=3`) yolcularınki (%24.2) en düşüktür. Sosyo-ekonomik durumun hayatta kalmada önemli bir faktör olduğu görülmektedir.
* **Korelasyonlar:** Hayatta kalma (`Survived`) ile en güçlü pozitif korelasyon 'Sex' (%54.3) ile, en güçlü negatif korelasyon ise 'Pclass' (%-33.8) iledir. Bu, kadın olmanın ve üst sınıfta bulunmanın hayatta kalma şansını artırdığını göstermektedir.

---

Bu proje, NumPy ve Pandas kullanarak temel bir veri analizi projesinin nasıl yapılabileceğini göstermektedir. Kodlar ve adımlar `sample.ipynb` dosyasında detaylı olarak incelenebilir.